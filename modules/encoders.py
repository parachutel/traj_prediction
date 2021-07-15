import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.latent import DiscreteLatent

from args import args

class Encoder(nn.Module):
    def __init__(self,
                 state_dim=6,
                 rel_state_dim=6,
                 edge_type_dim=4,
                 nhe_hidden_size=128,
                 ehe_hidden_size=128,
                 nfe_hidden_size=128,
                 masked_ehe=True,
                 device='cpu'):
        super().__init__()
        self.nhe_encoder = NodeHistroyEncoder(state_dim=state_dim,
                                              nhe_hidden_size=nhe_hidden_size,
                                              device=device)
        ehe_class = MaskedEdgeHistoryEncoder if masked_ehe else EdgeHistoryEncoder
        self.ehe_encoder = ehe_class(rel_state_dim=rel_state_dim,
                                     edge_type_dim=edge_type_dim,
                                     ehe_hidden_size=ehe_hidden_size,
                                     device=device)
        self.nfe_encoder = NodeFutureEncoder(state_dim=state_dim,
                                             nfe_hidden_size=nfe_hidden_size,
                                             device=device)
        
        self.x_size = self.nhe_encoder.output_size + self.ehe_encoder.output_size
        self.y_size = self.nfe_encoder.output_size

        self.latent_xy_input_mlp = nn.Linear(self.x_size + self.y_size, 
                                             args.latent_input_size)
        self.latent_x_input_mlp = nn.Linear(self.x_size, args.latent_input_size)
        self.latent = DiscreteLatent(latent_input_size=args.latent_input_size,
                                     n_latent_vars=args.n_latent_vars,
                                     latent_dim=args.latent_dim,
                                     kl_min=args.kl_min,
                                     device=device)
        self.device = device

    def forward(self, input_seqs, input_masks, input_edge_types, pred_seqs=None, mode='training'):
        nhe = self.nhe_encoder(input_seqs, mode)
        ehe = self.ehe_encoder(input_seqs, input_masks, input_edge_types, mode)
        x = torch.cat([nhe, ehe], dim=-1)
        # (bs, x_size), time dimension is summarized by encoders
        if mode == 'predict':
            return x
        else:
            assert pred_seqs is not None
            y = self.nfe_encoder(input_seqs, pred_seqs, mode)
            return x, y

    def q_z_xy(self, x, y, mode):
        xy = torch.cat([x, y], dim=-1)

        h_xy = F.relu(self.latent_xy_input_mlp(xy))
        h_xy = F.dropout(h_xy, 
                         p=args.mlp_dropout_prob,
                         training=(mode == 'training'))
        h_xy = self.latent.xy_to_latent(h_xy) # z_dim
        # self.latent.q_dist
        return self.latent.z_dist_from_hidden(h_xy, mode)


    def p_z_x(self, x, mode):
        h_x = F.relu(self.latent_x_input_mlp(x))
        h_x = F.dropout(h_x, 
                        p=args.mlp_dropout_prob,
                        training=(mode == 'training'))
        h_x = self.latent.x_to_latent(h_x) # z_dim
        # self.latent.p_dist
        return self.latent.z_dist_from_hidden(h_x, mode)


    def get_z_and_kl_qp(self, x, y, mode):
        # only used for training and eval
        if mode == 'training':
            num_samples = args.n_z_samples_training
        elif mode == 'eval':
            num_samples = args.n_z_samples_eval
        else:
            raise RuntimeError('Not for prediction')

        self.latent.q_dist = self.q_z_xy(x, y, mode)
        self.latent.p_dist = self.p_z_x(x, mode)
    
        z = self.latent.sample_q(num_samples, mode)

        if mode == 'training' and args.kl_exact:
            kl_obj = self.latent.kl_q_p()
        else:
            kl_obj = None

        return z, kl_obj

            

class NodeHistroyEncoder(nn.Module):
    def __init__(self,
                 state_dim=6,
                 nhe_hidden_size=128,
                 device='cpu'):
        super().__init__()
        self.nhe = nn.LSTM(input_size=state_dim,
                           hidden_size=nhe_hidden_size)
        # NOT batch_first, (seq, batch, feature)

        self.output_size = nhe_hidden_size
        self.device = device

    def forward(self, input_seqs, mode):
        # input_seqs.shape = (bs, seq, 3, 3, state_dim)
        node_state_histories = input_seqs[:, :, 1, 1] # center of the 3x3 grid
        # (bs, seq, state_dim)
        node_state_histories = node_state_histories.transpose(0, 1)
        # (seq, bs, state_dim)
        hiddens, (_, _) = self.nhe(node_state_histories)
        # (seq, bs, hidden_size)
        outputs = hiddens[-1] # (bs, hidden_size), last step hidden state
        outputs = F.dropout(outputs,
                            p=args.rnn_dropout_prob,
                            training=(mode == 'training'))
        return outputs


class EdgeHistoryEncoder(nn.Module):
    def __init__(self,
                 rel_state_dim=6,
                 edge_type_dim=4,
                 ehe_hidden_size=128,
                 device='cpu'):
        super().__init__()
        print('Using EdgeHistoryEncoder')
        self.edge_info_fusion = nn.Linear(rel_state_dim + edge_type_dim + 9, 128)
        self.ehe = nn.LSTM(input_size=128,
                           hidden_size=ehe_hidden_size)
        # NOT batch_first, (seq, batch, feature)
        # Single head self attention
        self.k = nn.Linear(ehe_hidden_size, 64)
        self.q = nn.Linear(ehe_hidden_size, 64)
        self.v = nn.Linear(ehe_hidden_size, 64)

        self.output_size = 64
        self.output_reshaper = nn.Linear(64, 16)
        self.output = nn.Linear(16 * 9, self.output_size)

        self.device = device

    def forward(self, input_seqs, input_masks, input_edge_types, mode):
        '''
            input_seqs.shape = (bs, seq, 3, 3, state_dim)
            input_edge_types.shape = (bs, seq, 3, 3, 4)
            input_masks NOT used
        '''
        bs, seq_len = input_seqs.shape[0], input_seqs.shape[1]
        
        # Location onehot embeddings
        loc_onehot = torch.eye(9).to(self.device) # (9, 9)
        loc_onehot = loc_onehot.unsqueeze(0).unsqueeze(0) # (1, 1, 9, 9)
        loc_onehot = loc_onehot.expand(bs, seq_len, -1, -1) # (bs, seq, 9, 9)
        loc_onehot = loc_onehot.reshape(bs, seq_len, 3, 3, 9) # (bs, seq, 3, 3, 9)
        
        edge_info = torch.cat([input_seqs, input_edge_types, loc_onehot], dim=-1)
        # (bs, seq, 3, 3, 4 + state_dim + 9)
        edge_info = edge_info.reshape(bs, seq_len, 9, -1)
        # (bs, seq, 9, 4 + state_dim + 9)
        edge_info = edge_info.transpose(0, 1)
        # (seq, bs, 9, 4 + state_dim + 9)
        edge_info = edge_info.reshape(seq_len, bs * 9, -1)
        # (seq, bs * 9, 4 + state_dim + 9)
        edge_info = self.edge_info_fusion(edge_info)
        # (seq, bs * 9, fusion_dim)
        edge_info, (_, _) = self.ehe(edge_info)
        # (seq, bs * 9, ehe_hidden_size)
        edge_info = edge_info[-1].reshape(bs, 9, -1) # take the last step
        # (bs, 9, ehe_hidden_size)

        edge_info = F.dropout(edge_info,
                              p=args.rnn_dropout_prob,
                              training=(mode == 'training'))
        
        # Single head self attention among the 9 adjacency grids
        key = self.k(edge_info) # (bs, 9, 64)
        query = self.q(edge_info) # (bs, 9, 64)
        value = self.v(edge_info) # (bs, 9, 64)

        query = query.transpose(1, 2) # (bs, 64, 9)
        att = key @ query / math.sqrt(64) # (bs, 9, 9)
        att = F.softmax(att, dim=-1) # (bs, 9, 9)

        edge_info = att @ value # (bs, 9, 64)

        edge_info = F.relu(self.output_reshaper(edge_info)) # (bs, 9, 16)
        edge_info = edge_info.reshape(bs, -1) # (bs, 9 * 16)
        edge_info = self.output(edge_info) # (bs, 64)

        edge_info = F.dropout(edge_info,
                              p=args.rnn_dropout_prob,
                              training=(mode == 'training'))

        return edge_info

class MaskedEdgeHistoryEncoder(nn.Module):
    def __init__(self,
                 rel_state_dim=6,
                 edge_type_dim=4,
                 ehe_hidden_size=128,
                 device='cpu'):
        super().__init__()
        print('Using MaskedEdgeHistoryEncoder')
        self.edge_info_fusion = nn.Linear(rel_state_dim + edge_type_dim + 9, 64)

        self.n_heads = 4
        self.att_size = 64
        assert self.att_size % self.n_heads == 0
        self.head_size = self.att_size // self.n_heads
        self.k = nn.Linear(rel_state_dim, self.att_size)
        self.q = nn.Linear(64, self.att_size)
        self.v = nn.Linear(64, self.att_size)

        self.ehe = nn.LSTM(input_size=self.att_size,
                           hidden_size=ehe_hidden_size)
        # NOT batch_first, (seq, batch, feature)

        self.output_size = 64
        self.output = nn.Linear(ehe_hidden_size, self.output_size)

        self.device = device

    def forward(self, input_seqs, input_masks, input_edge_types, mode):
        '''
            input_seqs.shape = (bs, seq, 3, 3, state_dim)
            input_edge_types.shape = (bs, seq, 3, 3, 4)
            input_masks.shape = (bs, seq, 3, 3)
        '''
        bs, seq_len = input_seqs.shape[0], input_seqs.shape[1]
        
        # Location onehot embeddings
        loc_onehot = torch.eye(9).to(self.device) # (9, 9)
        loc_onehot = loc_onehot.unsqueeze(0).unsqueeze(0) # (1, 1, 9, 9)
        loc_onehot = loc_onehot.expand(bs, seq_len, -1, -1) # (bs, seq, 9, 9)
        loc_onehot = loc_onehot.reshape(bs, seq_len, 3, 3, 9) # (bs, seq, 3, 3, 9)
        
        edge_info = torch.cat([input_seqs, input_edge_types, loc_onehot], dim=-1)
        # (bs, seq, 3, 3, 4 + state_dim + 9)
        edge_info = self.edge_info_fusion(edge_info)
        # (bs, seq, 3, 3, fusion_dim)
        edge_info = F.dropout(edge_info,
                              p=args.rnn_dropout_prob,
                              training=(mode == 'training'))
        self_info = input_seqs[:, :, 1, 1]
        # (bs, seq, state_dim)

        # Reshaping:
        edge_info = edge_info.reshape(bs, seq_len, 9, -1)
        # (bs, seq, 9, fusion_dim)
        self_info = self_info.unsqueeze(-2).expand(-1, -1, 9, -1)
        # (bs, seq, 9, state_dim)

        # Attention of self towards others:
        _att_shape = (bs, seq_len, 9, self.n_heads, self.head_size)
        key = self.k(self_info).reshape(_att_shape).transpose(2, 3)
        query = self.q(edge_info).reshape(_att_shape).transpose(2, 3)
        value = self.v(edge_info).reshape(_att_shape).transpose(2, 3)
        # (bs, seq, 9, att_size) -> (bs, seq, 9, n_heads, head_size) -> (bs, seq, n_heads, 9, head_size)
        
        # Create dot product shapes
        key = key.unsqueeze(-2)
        # (bs, seq, n_heads, 9, 1, head_size), 9 repeating self_info encoding
        query = query.unsqueeze(-1)
        # (bs, seq, n_heads, 9, head_size, 1), 9 distinct edge_info encoding

        att = (key @ query).squeeze() / math.sqrt(self.head_size)
        # (bs, seq, n_heads, 9), 9 weights of other nodes wrt self node


        # Mask out the center grid and other empty nodes (where input_masks = 0)
        inv_input_masks = (1 - input_masks.reshape(bs, seq_len, -1)).type(torch.BoolTensor)
        # (bs, seq, 9)
        inv_input_masks = inv_input_masks.unsqueeze(-2)
        # (bs, seq, 1, 9)
        inv_input_masks = inv_input_masks.expand(-1, -1, self.n_heads, -1)
        # (bs, seq, n_heads, 9), same mask for each head

        # print(att[0,0])
        # print(input_edge_types[0,0])
        # print(inv_input_masks[0,0])

        att.masked_fill_(inv_input_masks.to(self.device), -1e10) # -float('inf') is unstable
        att = F.softmax(att, dim=-1) 
        # (bs, seq, n_heads, 9)

        # print(att[0,0], att[0,0].sum())

        att = att.unsqueeze(-1) # prepare for elementwise product with value
        # (bs, seq, n_heads, 9, 1)
        edge_info = att * value 
        # (bs, seq, n_heads, 9, head_size)
        edge_info = edge_info.contiguous().sum(-2) 
        # (bs, seq, n_heads, head_size), att weighted sum for each head
        edge_info = edge_info.reshape(bs, seq_len, -1)
        # (bs, seq, att_size), att_size = hidden_size
        
        # LSTM
        edge_info = edge_info.transpose(0, 1)
        # (seq, bs, hidden_size)
        edge_info, (_, _) = self.ehe(edge_info)
        # (seq, bs, ehe_hidden_size)
        edge_info = edge_info[-1] # take the last step hidden
        # (bs, ehe_hidden_size)

        edge_info = F.dropout(edge_info,
                              p=args.rnn_dropout_prob,
                              training=(mode == 'training'))
        
        edge_info = self.output(edge_info) # (bs, 64)

        return edge_info


class NodeFutureEncoder(nn.Module):
    def __init__(self,
                 state_dim=6,
                 nfe_hidden_size=128,
                 device='cpu'):
        super().__init__()
        self.nfe = nn.LSTM(input_size=state_dim,
                           hidden_size=nfe_hidden_size,
                           bidirectional=True)
        # NOT batch_first, (seq, batch, feature)
        self.initial_h = nn.Linear(state_dim, nfe_hidden_size)
        self.initial_c = nn.Linear(state_dim, nfe_hidden_size)
        self.output_size = 64
        self.output = nn.Linear(2 * nfe_hidden_size, self.output_size)

        self.device = device

    def forward(self, input_seqs, pred_seqs, mode):
        # input_seqs.shape = (bs, seq, 3, 3, state_dim)
        # pred_seqs.shape = (bs, seq, 3, 3, state_dim)
        node_state_futures = pred_seqs[:, :, 1, 1] # center of the 3x3 grid
        # (bs, seq, state_dim)
        node_state_futures = node_state_futures.transpose(0, 1)
        # (seq, bs, state_dim)
        curr_state = input_seqs[:, -1, 1, 1] # all batches, last step, center grid
        # (bs, state_dim)
        h0 = self.initial_h(curr_state) # (bs, nfe_hidden_size)
        h0 = torch.stack([h0, torch.zeros_like(h0, device=self.device)], dim=0) # (2, bs, nfe_hidden_size)
        c0 = self.initial_c(curr_state)# (bs, nfe_hidden_size)
        c0 = torch.stack([c0, torch.zeros_like(c0, device=self.device)], dim=0) # (2, bs, nfe_hidden_size)
        hiddens, (_, _) = self.nfe(node_state_futures, (h0, c0))
        # (seq, bs, hidden_size * 2)
        future_encoding = F.dropout(hiddens[-1],
                                    p=args.rnn_dropout_prob,
                                    training=(mode == 'training'))
        future_encoding = self.output(future_encoding) # (bs, 64)
        return future_encoding