import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: add dropout

class Encoders(nn.Module):
    def __init__(self,
                 state_dim=6,
                 rel_state_dim=6,
                 edge_type_dim=4,
                 nhe_hidden_size=128,
                 ehe_hidden_size=128,
                 nfe_hidden_size=128,
                 device='cpu'):
        super().__init__()
        self.nhe_encoder = NodeHistroyEncoder(state_dim=state_dim,
                                              nhe_hidden_size=nhe_hidden_size,
                                              device=device)
        self.ehe_encoder = EdgeHistoryEncoder(rel_state_dim=rel_state_dim,
                                              edge_type_dim=edge_type_dim,
                                              ehe_hidden_size=ehe_hidden_size,
                                              device=device)
        self.nfe_encoder = NodeFutureEncoder(state_dim=state_dim,
                                             nfe_hidden_size=nfe_hidden_size,
                                             device=device)
        self.x_size = self.nhe_encoder.output_size + self.ehe_encoder.output_size
        self.y_size = self.nfe_encoder.output_size
        self.device = device

    def forward(self, input_seqs, input_edge_types, pred_seqs=None):
        nhe = self.nhe_encoder(input_seqs)
        ehe = self.ehe_encoder(input_seqs, input_edge_types)
        x = torch.cat([nhe, ehe], dim=-1)
        # (bs, x_size), time dimension is summarized by encoders
        if self.training:
            assert pred_seqs is not None
            y = self.nfe_encoder(input_seqs, pred_seqs)
            return x, y
        else:
            return x

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

    def forward(self, input_seqs):
        # input_seqs.shape = (bs, seq, 3, 3, state_dim)
        node_state_histories = input_seqs[:, :, 1, 1] # center of the 3x3 grid
        # (bs, seq, state_dim)
        node_state_histories = node_state_histories.transpose(0, 1)
        # (seq, bs, state_dim)
        hiddens, (_, _) = self.nhe(node_state_histories)
        # (seq, bs, hidden_size)
        return hiddens[-1] # (bs, hidden_size), last step hidden state


class EdgeHistoryEncoder(nn.Module):
    def __init__(self,
                 rel_state_dim=6,
                 edge_type_dim=4,
                 ehe_hidden_size=128,
                 device='cpu'):
        super().__init__()
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

    def forward(self, input_seqs, input_edge_types):
        '''
            input_seqs.shape = (bs, seq, 3, 3, state_dim)
            input_edge_types.shape = (bs, seq, 3, 3, 4)
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
        
        # Single head self attention among the 9 adjacency grids
        key = self.k(edge_info) # (bs, 9, 64)
        query = self.q(edge_info) # (bs, 9, 64)
        value = self.v(edge_info) # (bs, 9, 64)

        query = query.transpose(1, 2) # (bs, 64, 9)
        att = key @ query / math.sqrt(64) # (bs, 9, 9)
        # Mask out the center grid (indexed 4)
        # tgt_node_mask = 
        # att.masked_fill_(tgt_node_mask, -float('inf'))
        att = F.softmax(att, dim=-1) # (bs, 9, 9)

        edge_info = att @ value # (bs, 9, 64)

        edge_info = F.relu(self.output_reshaper(edge_info)) # (bs, 9, 16)
        edge_info = edge_info.reshape(bs, -1) # (bs, 9 * 16)
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

    def forward(self, input_seqs, pred_seqs):
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
        future_encoding = self.output(hiddens[-1]) # (bs, 64)
        return future_encoding