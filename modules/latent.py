import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np

def all_one_hot_combinations(N, K):
    return np.eye(K).take(np.reshape(np.indices([K] * N), [N, -1]).T, axis=0).reshape(-1, N * K)

class DiscreteLatent(nn.Module):
    def __init__(self, 
                 latent_input_size=128,
                 n_latent_vars=2,
                 latent_dim=5,
                 kl_min=0.07, 
                 device='cpu'):
        super().__init__()

        self.N = n_latent_vars
        self.K = latent_dim
        self.z_dim = n_latent_vars * latent_dim # latent_dim
        self.kl_min = kl_min

        self.temp = None            # filled in by setup_hyperparams_annealing in Predictor
        self.z_logit_clip = None    # filled in by setup_hyperparams_annealing in Predictor
        self.p_dist = None
        self.q_dist = None

        self.xy_to_latent = nn.Linear(latent_input_size, self.z_dim)
        self.x_to_latent = nn.Linear(latent_input_size, self.z_dim)

        self.device = device


    def z_dist_from_hidden(self, h, mode):
        # h.shape = (bs, z_dim)
        logits_separated = h.reshape(-1, self.N, self.K)
        logits_separated_mean_zero = logits_separated - torch.mean(logits_separated, dim=-1, keepdim=True)
        if self.z_logit_clip is not None and mode == 'training':
            c = self.z_logit_clip
            logits = torch.clamp(logits_separated_mean_zero, min=-c, max=c)
        else:
            logits = logits_separated_mean_zero
        
        if logits.shape[0] == 1:
            logits = torch.squeeze(logits, dim=0)
        
        return td.OneHotCategorical(logits=logits) # (bs, N, K)


    def sample_q(self, k, mode):
        '''
            k is the number of samples
            Not used for prediction
        '''
        if mode == 'training':
            z_dist = td.RelaxedOneHotCategorical(self.temp, logits=self.q_dist.logits)
            z_NK = z_dist.rsample((k, )) # make backprop gradient passable
        elif mode == 'eval':
            z_NK = self.q_dist.sample((k, ))
        return z_NK.reshape(k, -1, self.z_dim) # (n_samples, bs, z_dim)


    def sample_p(self, k, mode, most_likely=False):
        if mode == 'predict' and self.K ** self.N < 100 and k == 0:
            bs = self.p_dist.probs.size()[0]
            z_NK = torch.from_numpy(
                all_one_hot_combinations(self.N, self.K)).to(self.device).repeat(1, bs)
            k = self.K ** self.N # total possibilities of z

        elif most_likely:
            # z-best
            # Sampling the most likely z from p(z|x).
            # self.p_dist.event_shape returns the shape of a single sample (without batching)
            # self.p_dist.event_shape = (K,)
            eye_mat = torch.eye(self.p_dist.event_shape[-1], device=self.device)
            # eye_mat.shape = (K, K)
            # self.p_dist.probs.shape = (bs, N, K)
            argmax_idxs = torch.argmax(self.p_dist.probs, dim=-1)
            # (bs, N), the most probable latent var
            argmax_eye_mat = eye_mat[argmax_idxs]
            # (bs, N, K)
            argmax_eye_mat = torch.unsqueeze(argmax_eye_mat, dim=0)
            # (1, bs, N, K)
            z_NK = argmax_eye_mat.expand(k, -1, -1, -1)
            # (k, bs, N, K)
        else:
            # z-full
            # Normally use this naive sampling method
            z_NK = self.p_dist.sample((k, ))
            # (k, bs, N, K)

        return z_NK.reshape(k, -1, self.N * self.K)


    def kl_q_p(self):
        # self.q_dist.probs.shape = (bs, N, K)
        kl_separated = td.kl.kl_divergence(self.q_dist, self.p_dist)
        # kl_separated.shape = (bs, N)
        
        if len(kl_separated.shape) < 2:
            kl_separated = torch.unsqueeze(kl_separated, dim=0)
            
        kl_minibatch = torch.mean(kl_separated, dim=0, keepdim=True)

        if self.kl_min > 0:
            kl_lower_bounded = torch.clamp(kl_minibatch, min=self.kl_min)
            kl = torch.sum(kl_lower_bounded)
        else:
            kl = torch.sum(kl_minibatch)
        
        return kl


    def q_log_prob(self, z):
        k = z.size()[0]
        z_NK = torch.reshape(z, [k, -1, self.N, self.K])
        return torch.sum(self.q_dist.log_prob(z_NK), dim=2)


    def p_log_prob(self, z):
        k = z.size()[0]
        z_NK = torch.reshape(z, [k, -1, self.N, self.K])
        return torch.sum(self.p_dist.log_prob(z_NK), dim=2)


    def get_p_dist_probs(self):
        return self.p_dist.probs