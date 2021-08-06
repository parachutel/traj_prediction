import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import sys
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../')

from modules.encoders import Encoder
from modules.decoder import Decoder
from modules.latent import all_one_hot_combinations

from model_utils.annealer import setup_hyperparams_annealing
from args import args

class Predictor(nn.Module):
    def __init__(self, 
                 state_dim=args.state_dim,
                 rel_state_dim=args.state_dim,
                 pred_dim=args.pred_dim,
                 edge_type_dim=args.n_edge_types,
                 nhe_hidden_size=args.nhe_hidden_size,
                 ehe_hidden_size=args.ehe_hidden_size,
                 nfe_hidden_size=args.nfe_hidden_size,
                 decoder_hidden_size=args.decoder_hidden_size,
                 gmm_components=args.gmm_components,
                 log_sigma_min=args.log_sigma_min,
                 log_sigma_max=args.log_sigma_max,
                 log_p_yt_xz_max=args.log_p_yt_xz_max,
                 kl_weight=args.kl_weight,
                 masked_ehe=args.masked_ehe,
                 device='cpu'):

        super().__init__()
        
        self.encoder = Encoder(state_dim=state_dim,
                               rel_state_dim=rel_state_dim,
                               edge_type_dim=edge_type_dim,
                               nhe_hidden_size=nhe_hidden_size,
                               ehe_hidden_size=ehe_hidden_size,
                               nfe_hidden_size=nfe_hidden_size,
                               masked_ehe=masked_ehe,
                               device=device)
        
        self.decoder = Decoder(x_size=self.encoder.x_size,
                               z_size=self.encoder.latent.z_dim,
                               pred_dim=pred_dim,
                               decoder_hidden_size=decoder_hidden_size,
                               gmm_components=gmm_components,
                               log_sigma_min=log_sigma_min,
                               log_sigma_max=log_sigma_max,
                               log_p_yt_xz_max=log_p_yt_xz_max,
                               device=device)

        self.kl_weight = kl_weight

        self.device = device

        self.schedulers = []
        self.dummy_optimizers = []
        self.annealed_var_names = []
        setup_hyperparams_annealing(self)

    def forward(self, input_seq, input_masks, input_edge_types):
        '''
            for exporting to onnx only
        '''
        x = self.encoder(input_seqs, input_masks, input_edge_types, mode='predict')
        self.encoder.latent.p_dist = self.encoder.p_z_x(x, 'predict')
        z = self.encoder.latent.sample_p(args.n_z_samples_pred, mode='predict', most_likely=True)
        log_pi_t, mu_t, log_sigma_t, corr_t, zx = self.decoder(x, z, input_seqs)
        return log_pi_t, mu_t, log_sigma_t, corr_t, zx


    def get_training_loss(self, input_seq, input_masks, input_edge_types, pred_seq):
        mode = 'training'

        x, y = self.encoder(input_seq, input_masks, input_edge_types, pred_seq, mode)
        z, kl = self.encoder.get_z_and_kl_qp(x, y, mode)

        log_p_y_xz = self.decoder.log_p_y_xz(x, z, args.n_z_samples_training, input_seq, pred_seq,
            mode=mode, sample_model_during_decoding=args.sample_model_during_dec) # (bs, pred_dim)

        if np.abs(args.alpha - 1.0) < 1e-3 and not args.use_iwae:
            log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0) # (pred_dim,)
            log_likelihood = torch.mean(log_p_y_xz_mean) # (1,)
            ELBO = log_likelihood - self.kl_weight * kl

            # Debug gradients
            # torch.set_printoptions(profile="full")
            # llh_grad = torch.autograd.grad(outputs=log_likelihood, 
            #     inputs=self.encoder.latent_xy_input_mlp.weight, 
            #     grad_outputs=torch.ones(log_likelihood.size()),
            #     retain_graph=True)
            # print('llh', log_likelihood)
            # print('llh_grad', llh_grad[0].mean())

            # kl_grad = torch.autograd.grad(outputs=kl, 
            #     inputs=self.encoder.latent.x_to_latent.weight, 
            #     grad_outputs=torch.ones(kl.size()),
            #     retain_graph=True)
            # print('kl_grad', kl_grad)

            loss = -ELBO

        else:
            log_q_z_xy = self.encoder.latent.q_log_prob(z)
            log_p_z_x = self.encoder.latent.p_log_prob(z)
            a = args.alpha
            log_pp_over_q = log_p_y_xz + log_p_z_x - log_q_z_xy
            log_likelihood = (torch.mean(torch.logsumexp(log_pp_over_q * (1. - a), dim=0)) - \
                              torch.log(args.n_z_samples_training)) / (1. - a)
            loss = -log_likelihood

        return loss

    def get_eval_loss(self, input_seq, input_masks, input_edge_types, pred_seq,
                      compute_naive=True,
                      compute_exact=True):
        mode = 'eval'

        x, y = self.encoder(input_seq, input_masks, input_edge_types, pred_seq, mode)

        ### Importance sampled NLL estimate
        z, _ = self.encoder.get_z_and_kl_qp(x, y, mode)
        log_p_y_xz = self.decoder.log_p_y_xz(x, z, args.n_z_samples_eval, input_seq, pred_seq, mode)
        log_q_z_xy = self.encoder.latent.q_log_prob(z)
        log_p_z_x = self.encoder.latent.p_log_prob(z)
        log_likelihood = torch.mean(torch.logsumexp(log_p_y_xz + log_p_z_x - log_q_z_xy, dim=0)) - \
                         torch.log(torch.tensor(
                                args.n_z_samples_eval, dtype=torch.float, device=self.device))
        nll_q_is = -log_likelihood

        ### Naive sampled NLL estimate
        nll_p = torch.tensor(np.nan)
        if compute_naive:
            z = self.encoder.latent.sample_p(args.n_z_samples_eval, mode, most_likely=False)
            log_p_y_xz = self.decoder.log_p_y_xz(x, z, args.n_z_samples_eval, input_seq, pred_seq, mode)
            log_likelihood_p = torch.mean(torch.logsumexp(log_p_y_xz, dim=0)) - \
                               torch.log(torch.tensor(
                                    args.n_z_samples_eval, dtype=torch.float, device=self.device))
            nll_p = -log_likelihood_p

        ### Exact NLL
        nll_exact = torch.tensor(np.nan)
        if compute_exact:
            K, N = args.pred_dim, args.latent_dim
            n_z_samples = K ** N
            if n_z_samples < 50:
                nbs = x.shape[0]
                z_raw = torch.from_numpy(
                        all_one_hot_combinations(N, K).astype(np.float32)
                    ).to(self.device).repeat(1, nbs)
                z = z_raw.reshape(n_z_samples, -1, N * K)
                log_p_y_xz = self.decoder.log_p_y_xz(x, z, n_z_samples, input_seq, pred_seq, mode)
                log_p_z_x = self.encoder.latent.p_log_prob(z)
                exact_log_likelihood = torch.mean(torch.logsumexp(log_p_y_xz + log_p_z_x, dim=0))
                nll_exact = -exact_log_likelihood

        return nll_q_is, nll_p, nll_exact


    def predict(self, input_seqs, input_masks, input_edge_types, num_samples, most_likely=False):
        mode = 'predict'

        x = self.encoder(input_seqs, input_masks, input_edge_types, pred_seqs=None, mode=mode)
        # (bs, encoder.x_size)
        self.encoder.latent.p_dist = self.encoder.p_z_x(x, mode)
        z_p_samples = self.encoder.latent.sample_p(num_samples, mode, most_likely=most_likely)
        y_dist, sampled_future = self.decoder.p_y_xz(x, z_p_samples, num_samples, input_seqs, 
                                                     n_pred_steps=args.n_pred_steps,
                                                     mode=mode)
        return sampled_future, z_p_samples


if __name__ == '__main__':
    predictor = Predictor()
    in_seq_len = args.input_seconds * args.highd_frame_rate
    bs = 1
    input_seqs = torch.rand(bs, in_seq_len, 3, 3, args.state_dim)
    input_masks = torch.rand(bs, in_seq_len, 3, 3)
    input_edge_types = torch.rand(bs, in_seq_len, 3, 3, args.n_edge_types)
    input_names = ['input_seq', 'input_mask', 'input_edge_types']
    dummy_inputs = (input_seqs, input_masks, input_edge_types)

    torch.onnx.export(predictor, dummy_inputs, '../save/tmp/predictor.onnx', 
        verbose=False, input_names=input_names, 
        output_names=['log_pi_t', 'mu_t', 'log_sigma_t', 'corr_t', 'zx'],
        opset_version=12)