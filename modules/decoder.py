import torch
import torch.nn as nn

from modules.gmm import GMMParams, GMM2D
from torch.distributions import Bernoulli

class Decoder(nn.Module):
    def __init__(self,
                 x_size,
                 z_size,
                 pred_dim=2,
                 decoder_hidden_size=64,
                 gmm_components=16,
                 log_sigma_min=-10,
                 log_sigma_max=10,
                 log_p_yt_xz_max=50,
                 decoding_sample_model_prob=0.5, # Not used
                 device='cpu',
                ):
        super().__init__()

        self.z_size = z_size
        self.initial_h = nn.Linear(z_size + x_size, decoder_hidden_size)
        self.initial_c = nn.Linear(z_size + x_size, decoder_hidden_size)

        self.lstm_cell = nn.LSTMCell(input_size=z_size + x_size + pred_dim,
                                     hidden_size=decoder_hidden_size)
        
        self.gmm_params = GMMParams(decoder_hidden_size=decoder_hidden_size,
                                    gmm_components=gmm_components,
                                    pred_dim=pred_dim)
        
        self.gmm_components = gmm_components
        self.pred_dim = pred_dim
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.log_p_yt_xz_max = log_p_yt_xz_max
        self.decoding_sample_model_prob = None # set by annealer
        self.device = device

    def p_y_xz(self, x, z_stacked, n_z_samples, input_seqs, pred_seqs=None,
               n_pred_steps=None, mode='training', sample_model_during_decoding=True):

        # x is the output of history encoders
        # input_seqs (bs, input_seq_len, 3, 3, state_dim)
        # pred_seqs (bs, pred_seq_len, 3, 3, state_dim)

        if pred_seqs is not None:
            n_pred_steps = pred_seqs.shape[1]
            tgt_future = pred_seqs[:, :, 1, 1, 2:4] # (x_dot, y_dot), (bs, pred_seq_len, pred_dim)

        z = z_stacked.reshape(-1, z_stacked.shape[-1]) # (bs * n_z_samples, z_dim)
        zx = torch.cat([z, x.repeat(n_z_samples, 1)], dim=1) # (bs * n_z_samples, z_dim + x_dim)
        initial_state = (self.initial_h(zx), self.initial_c(zx)) # (bs * n_z_samples, decoder_hidden_size)

        rnn_state = initial_state # hidden 
        tgt_prediction_present = input_seqs[:, -1, 1, 1, 2:4] # (x_dot, y_dot)
        # x_dot, y_dot (idx 2 and 3) of target traj (grid 1, 1) at current step (t=-1 of input_seqs), (bs, pred_dim)
        input_ = torch.cat([zx, tgt_prediction_present.repeat(n_z_samples, 1)], dim=-1)
        # (bs * n_z_samples, z_dim + x_dim + pred_dim)

        if mode in ['training', 'eval']:
            assert pred_seqs is not None
            if sample_model_during_decoding and mode == 'training':
                log_pis, mus, log_sigmas, corrs = [], [], [], []
                for t in range(n_pred_steps):
                    h_state, c_state = self.lstm_cell(input_, rnn_state)
                    log_pi_t, mu_t, log_sigma_t, corr_t = self.gmm_params(h_state)
                    y_t = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t,
                                self.log_sigma_min, self.log_sigma_max, self.device).sample()
                    # (bs * n_z_samples, pred_dim)
    
                    mask = Bernoulli(probs=self.decoding_sample_model_prob).sample((y_t.shape[0], 1))
                    
                    tgt_future_t = tgt_future[:, t, :] # (bs, pred_dim), real future
                    y_t = mask * y_t + (1 - mask) * (tgt_future_t.repeat(n_z_samples, 1))
                    # (bs * n_z_samples, pred_dim)

                    log_pis.append(log_pi_t)
                    mus.append(mu_t)
                    log_sigmas.append(log_sigma_t)
                    corrs.append(corr_t)

                    input_ = torch.cat([zx, y_t], dim=1)
                    # (bs * n_z_samples, z_dim + hx_dim + pred_dim)
                    rnn_state = (h_state, c_state)

                log_pis = torch.stack(log_pis, dim=1)
                mus = torch.stack(mus, dim=1)
                log_sigmas = torch.stack(log_sigmas, dim=1)
                corrs = torch.stack(corrs, dim=1)

            else:
                zx_with_time_dim = zx.unsqueeze(dim=1) # (bs * n_z_samples, 1, z_dim + hx_dim)
                zx_time_tiled = zx_with_time_dim.repeat(1, n_pred_steps, 1)
                # (bs * n_z_samples, n_pred_steps, z_dim + hx_dim)

                decode_inputs = torch.cat([
                    tgt_prediction_present.unsqueeze(dim=1), 
                    tgt_future[:, :n_pred_steps - 1, :]], dim=1)
                # (bs, 1 + pred_seq_len - 1, pred_dim)
                # Fully using true future as inputs, NOT for predict
                
                inputs = torch.cat([zx_time_tiled, decode_inputs.repeat(n_z_samples, 1, 1)], dim=2)
                # (bs * n_z_samples, pred_seq_len, pred_dim)
                outputs = []
                for t in range(n_pred_steps):
                    h_state, c_state = self.lstm_cell(inputs[:, t, :], rnn_state)
                    outputs.append(h_state)
                    rnn_state = (h_state, c_state)
    
                outputs = torch.stack(outputs, dim=1) # (bs * n_z_samples, pred_seq_len, pred_dim)
                log_pis, mus, log_sigmas, corrs = self.gmm_params(outputs)
    
        else: # mode == 'predict'
            log_pis, mus, log_sigmas, corrs, y = [], [], [], [], []
            for t in range(n_pred_steps):
                h_state, c_state = self.lstm_cell(input_, rnn_state)
                log_pi_t, mu_t, log_sigma_t, corr_t = self.gmm_params(h_state)
                
                y_t = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t,
                            self.log_sigma_min, self.log_sigma_max, self.device).sample()
                
                log_pis.append(log_pi_t)
                mus.append(mu_t)
                log_sigmas.append(log_sigma_t)
                corrs.append(corr_t)
                y.append(y_t)

                input_ = torch.cat([zx, y_t], dim=1)
                rnn_state = (h_state, c_state)

            log_pis = torch.stack(log_pis, dim=1)
            mus = torch.stack(mus, dim=1)
            log_sigmas = torch.stack(log_sigmas, dim=1)
            corrs = torch.stack(corrs, dim=1)
            sampled_future = torch.stack(y, dim=1).reshape(n_z_samples, -1, n_pred_steps, self.pred_dim)

        #
        gmm_c = self.gmm_components
        y_dist = GMM2D(log_pis.reshape(n_z_samples, -1, n_pred_steps, gmm_c),
                       mus.reshape(n_z_samples, -1, n_pred_steps, gmm_c * self.pred_dim),
                       log_sigmas.reshape(n_z_samples, -1, n_pred_steps, gmm_c * self.pred_dim),
                       corrs.reshape(n_z_samples, -1, n_pred_steps, gmm_c),
                       self.log_sigma_min, self.log_sigma_max, self.device)

        if mode == 'predict':
            return y_dist, sampled_future
        else:
            return y_dist

    def log_p_y_xz(self, x, z, n_z_samples, input_seqs, pred_seqs=None, n_pred_steps=None, 
                   mode='training', sample_model_during_decoding=True):
        y_dist = self.p_y_xz(x, z, n_z_samples, input_seqs, pred_seqs, n_pred_steps, mode=mode,
                             sample_model_during_decoding=sample_model_during_decoding)
        # y_dist.sample (n_z_samples, bs, n_pred_steps, pred_dim)
        # True target future
        true_tgt_future = pred_seqs[:, :, 1, 1, 2:4] # (x_dot, y_dot), (bs, n_pred_steps, 2)
        # log_p_yt_xz = torch.clamp(y_dist.log_prob(true_tgt_future), max=self.log_p_yt_xz_max)
        log_p_yt_xz = y_dist.log_prob(true_tgt_future)
        log_p_y_xz = log_p_yt_xz.sum(dim=2) # sum through time dimension, (bs, 2)
        return log_p_y_xz