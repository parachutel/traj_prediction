import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical

def to_one_hot(labels, n_labels, device):
    return torch.eye(n_labels, device=device)[labels]

def reshape_to_components(tensor, GMM_c):
    return tensor.reshape(tensor.shape[:-1] + (GMM_c, -1))

class GMMParams(nn.Module):
    def __init__(self,
                 decoder_hidden_size=64,
                 gmm_components=16,
                 pred_dim=2 # predicting x_dot and y_dot
                ):
        super().__init__()

        self.get_log_pis = nn.Linear(decoder_hidden_size, gmm_components)
        self.get_mus = nn.Linear(decoder_hidden_size, gmm_components * pred_dim)
        self.get_log_sigmas = nn.Linear(decoder_hidden_size, gmm_components * pred_dim)
        self.get_corrs = nn.Linear(decoder_hidden_size, gmm_components)

    def forward(self, x):
        log_pis = self.get_log_pis(x)
        mus = self.get_mus(x)
        log_sigmas = self.get_log_sigmas(x)
        corrs = torch.tanh(self.get_corrs(x))
        return log_pis, mus, log_sigmas, corrs


class GMM2D(object):
    def __init__(self, log_pis, mus, log_sigmas, corrs, 
                 clip_lo=-10, clip_hi=10, device='cpu'):
        self.device = device

        # GMM_c: GMM components
        # pis: [..., GMM_c]
        # mus: [..., GMM_c * 2]
        # sigmas: [..., GMM_c * 2]
        # corrs: [..., GMM_c]
        GMM_c = log_pis.shape[-1]

        # Sigma = [s1^2    p*s1*s2      L = [s1   0
        #          p*s1*s2 s2^2 ]            p*s2 sqrt(1-p^2)*s2]
        log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)
        mus = reshape_to_components(mus, GMM_c) # [..., GMM_c, 2]
        log_sigmas = reshape_to_components(torch.clamp(log_sigmas, min=clip_lo, max=clip_hi), GMM_c)
        sigmas = torch.exp(log_sigmas) # [..., GMM_c, 2]
        one_minus_rho2 = 1 - corrs ** 2 # [..., GMM_c]

        self.L1 = sigmas * torch.stack([torch.ones_like(corrs, device=self.device), corrs], dim=-1)
        self.L2 = sigmas * torch.stack([torch.zeros_like(corrs, device=self.device), 
                                        torch.sqrt(one_minus_rho2)], dim=-1)

        self.batch_shape = log_pis.shape[:-1]
        self.GMM_c = GMM_c
        self.log_pis = log_pis                # [..., GMM_c]
        self.mus = mus                        # [..., GMM_c, 2]
        self.log_sigmas = log_sigmas          # [..., GMM_c, 2]
        self.sigmas = sigmas                  # [..., GMM_c, 2]
        self.corrs = corrs                    # [..., GMM_c]
        self.one_minus_rho2 = one_minus_rho2  # [..., GMM_c]
        self.cat = Categorical(logits=log_pis)


    def sample(self):
        MVN_samples = (self.mus 
                       + self.L1 * torch.unsqueeze(
                            torch.randn_like(self.corrs, device=self.device), dim=-1) # [..., GMM_c, 2]
                       + self.L2 * torch.unsqueeze(
                            torch.randn_like(self.corrs, device=self.device), dim=-1) # (manual 2x2 matmul)
                       )    
        cat_samples = self.cat.sample()
        selector = torch.unsqueeze(to_one_hot(cat_samples, self.GMM_c, self.device), dim=-1)
        return torch.sum(MVN_samples * selector, dim=-2)


    def log_prob(self, x):
        # x: [..., 2]
        x = torch.unsqueeze(x, dim=-2) # [..., 1, 2]
        dx = x - self.mus # [..., GMM_c, 2]
        z = (torch.sum((dx / self.sigmas) ** 2, dim=-1) -
             2 * self.corrs * torch.prod(dx, dim=-1) / torch.prod(self.sigmas, dim=-1)) # [..., GMM_c]
        component_log_p = -(torch.log(self.one_minus_rho2) + 2 * torch.sum(self.log_sigmas, dim=-1) +
                            z / self.one_minus_rho2 +
                            2 * np.log(2 * np.pi)) / 2
        return torch.logsumexp(self.log_pis + component_log_p, dim=-1)


    