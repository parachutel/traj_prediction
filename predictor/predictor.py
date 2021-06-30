import torch
import torch.nn as nn
import nn.functional as F

from modules.encoders import Encoders
from modules.latent import DiscreteLatent
from modules.decoder import Decoder


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = Encoders(state_dim=state_dim,
                                 rel_state_dim=rel_state_dim,
                                 edge_type_dim=edge_type_dim,
                                 nhe_hidden_size=nhe_hidden_size,
                                 ehe_hidden_size=ehe_hidden_size,
                                 nfe_hidden_size=nfe_hidden_size,
                                 device=device)
        self.latent = DiscreteLatent()
        self.decoder = Decoder()

        self.device = device

    def q_z_xy(self, x, y):
        xy = torch.cat([x, y], dim=1)
        # print('q_z_xy/xy', xy)

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            dense = self.node_modules[self.node.type + '/q_z_xy']
            h = F.dropout(F.relu(dense(xy)), 
                          p=1.-self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = xy

        to_latent = self.node_modules[self.node.type + '/hxy_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)

    def p_z_x(self, x):
        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            dense = self.node_modules[self.node.type + '/p_z_x']
            h = F.dropout(F.relu(dense(x)),
                          p=1.-self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = x

        to_latent = self.node_modules[self.node.type + '/hx_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)


    def get_latent(self, x, y):
        self.latent.q_dist = self.q_z_xy(x, y)
        self.latent.p_dist = self.p_z_x(x)

        z = self.latent.sample_q(self.num_latent_samples)

        if self.training and self.kl_exact:
            kl_obj = self.latent.kl_q_p()
        else:
            kl_obj = None

        return z, kl_obj


    
    def get_training_loss(self):
        return loss

    def get_eval_loss(self):
        return loss