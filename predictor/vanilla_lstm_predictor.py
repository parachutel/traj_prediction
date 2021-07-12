import torch
import torch.nn as nn
import torch.nn.functional as F

from args import args

class VanillaLSTMPredictor(nn.Module):

    def __init__(self,
                 state_dim=6,
                 hidden_size=32,
                 pred_dim=2,
                 device='cpu'):
        super().__init__()

        self.input_encoder = nn.Linear(state_dim, hidden_size)
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size)
        self.decoder_lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, pred_dim)

        self.device = device

    def predict(self, input_seqs, n_pred_steps=None):
        # input_seqs.shape = (bs, seq, 3, 3, state_dim)
        state_histories = input_seqs[:, :, 1, 1] # center of the 3x3 grid
        # (bs, seq, state_dim)
        state_histories = state_histories.transpose(0, 1)
        # (seq, bs, state_dim)
        state_histories = self.input_encoder(state_histories)
        # (seq, bs, hidden_size)
        hiddens, (h_n, c_n) = self.encoder_lstm(state_histories)
        # hiddens.shape = (seq, bs, hidden_size)

        rnn_state = (h_n.squeeze(0), c_n.squeeze(0))
        h_state = hiddens[-1] # (bs, hidden_size)
        preds = []
        if n_pred_steps is None:
            n_pred_steps = args.n_pred_steps
        for _ in range(n_pred_steps):
            h_state, c_state = self.decoder_lstm_cell(h_state, rnn_state)
            rnn_state = (h_state, c_state)
            pred = self.output(h_state)
            preds.append(pred)

        preds = torch.stack(preds, dim=1) # (bs, n_pred_steps, pred_dim)
        return preds

    def get_loss(self, input_seqs, pred_seqs):
        # pred_seqs (bs, pred_seq_len, 3, 3, state_dim)
        targets = pred_seqs[:, :, 1, 1, 2:4] 
        # (bs, pred_seq_len, pred_dim)

        n_pred_steps = pred_seqs.shape[1]

        preds = self.predict(input_seqs, n_pred_steps=n_pred_steps)
        # (bs, pred_seq_len, pred_dim)

        loss = F.mse_loss(preds, targets)
        return loss