import torch
import torch.nn as nn

from modules.gmm import GMMParams, GMM2D
from torch.distributions import Bernoulli

class Decoder(nn.Module):
    def __init__(self,
                 x_size,
                 z_size,
                 decoder_input_size,
                 decoder_hidden_size,
                 gmm_components=16,
                 pred_dim=2,
                 log_sigma_min=-10,
                 log_sigma_max=10,
                 log_p_yt_xz_max=0,
                 decoding_sample_model_prob=0.5,
                 device='cpu',
                ):
        super().__init__()

        self.z_size = z_size
        self.initial_h = nn.Linear(z_size + x_size, decoder_input_size)
        self.initial_c = nn.Linear(z_size + x_size, decoder_input_size)

        self.lstm_cell = nn.LSTMCell(input_size=decoder_input_size,
                                     hidden_size=decoder_hidden_size)
        
        self.gmm_params = GMMParams(decoder_hidden_size=decoder_hidden_size,
                                    gmm_components=gmm_components,
                                    pred_dim=pred_dim)
        
        self.gmm_components = gmm_components
        self.pred_dim = self.pred_dim
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.log_p_yt_xz_max = log_p_yt_xz_max
        self.decoding_sample_model_prob = decoding_sample_model_prob
        self.device = device

    def p_y_xz(self, x, z_stacked, TD, num_predicted_timesteps, num_samples, 
               sampling_decoding=True):

        z = z_stacked.reshape(z_stacked, (-1, self.size))
        zx = torch.cat([z, x.repeat(num_samples, 1)], dim=1)
        initial_state = (self.initial_h(zx), self.initial_c(zx))

        state = initial_state
        input_ = torch.cat([zx, TD['joint_present'].repeat(num_samples, 1)], dim=1)

        if self.training:
            if sampling_decoding:
                log_pis, mus, log_sigmas, corrs = [], [], [], []
                for t in range(num_predicted_timesteps):
                    h_state, c_state = self.lstm_cell(input_, state)
                    log_pi_t, mu_t, log_sigma_t, corr_t = self.gmm_params(h_state)
                    y_t = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t,
                                self.log_sigma_min, self.log_sigma_max, self.device).sample()
    
                    mask = td.Bernoulli(probs=self.decoding_sample_model_prob).sample((y_t.size()[0], 1))
    
    
                    y_t = mask * y_t + (1 - mask) * (TD[our_future][:,t,:].repeat(num_samples, 1))
                        
                    log_pis.append(log_pi_t)
                    mus.append(mu_t)
                    log_sigmas.append(log_sigma_t)
                    corrs.append(corr_t)
    
                    # if self.robot_node is not None:
                    #     dec_inputs = torch.cat([TD[robot_future][:,t,:].repeat(num_samples, 1), y_t], dim=1)
                    # else:
                    #     dec_inputs = y_t

                    dec_inputs = y_t
    
                    input_ = torch.cat([zx, dec_inputs], dim=1)
                    state = (h_state, c_state)
    
                
                log_pis = torch.stack(log_pis, dim=1)
                mus = torch.stack(mus, dim=1)
                log_sigmas = torch.stack(log_sigmas, dim=1)
                corrs = torch.stack(corrs, dim=1)
            
            else: # 
                zx_with_time_dim = zx.unsqueeze(dim=1)
                zx_time_tiled = zx_with_time_dim.repeat(1, num_predicted_timesteps, 1) 
                # if self.robot_node is not None:
                #     dec_inputs = torch.cat([
                #         TD["joint_present"].unsqueeze(dim=1),
                #         torch.cat([TD[robot_future][:,:num_predicted_timesteps-1,:], TD[our_future][:, :num_predicted_timesteps-1,:]], dim=2)
                #         ], dim=1)
                # else:
                #     dec_inputs = torch.cat([
                #         TD["joint_present"].unsqueeze(dim=1), 
                #         TD[our_future][:, :num_predicted_timesteps-1,:]
                #         ], dim=1)

                dec_inputs = torch.cat([
                    TD["joint_present"].unsqueeze(dim=1), 
                    TD[our_future][:, :num_predicted_timesteps-1,:]
                    ], dim=1)
                    
                inputs = torch.cat([zx_time_tiled, dec_inputs.repeat(num_samples, 1, 1)], dim=2)
                outputs = list()
                for j in range(num_predicted_timesteps):
                    h_state, c_state = cell(inputs[:, j, :], state)
                    outputs.append(h_state)
                    state = (h_state, c_state)
    
                outputs = torch.stack(outputs, dim=1)
                log_pis, mus, log_sigmas, corrs = self.gmm_params(outputs)
    
        else: # self.training = False, prediction/eval mode
            log_pis, mus, log_sigmas, corrs, y = [], [], [], [], []
            for j in range(num_predicted_timesteps):
                h_state, c_state = cell(input_, state)
                log_pi_t, mu_t, log_sigma_t, corr_t = self.gmm_params(h_state)
                
                y_t = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t,
                            self.log_sigma_min, self.log_sigma_max, self.device).sample()
                
                log_pis.append(log_pi_t)
                mus.append(mu_t)
                log_sigmas.append(log_sigma_t)
                corrs.append(corr_t)
                y.append(y_t)

                # if self.robot_node is not None:
                #     dec_inputs = torch.cat([TD[robot_future][:,j,:].repeat(num_samples, 1), y_t], dim=1)
                # else:
                #     dec_inputs = y_t

                dec_inputs = y_t

                input_ = torch.cat([zx, dec_inputs], dim=1)
                state = (h_state, c_state)

            log_pis = torch.stack(log_pis, dim=1)
            mus = torch.stack(mus, dim=1)
            log_sigmas = torch.stack(log_sigmas, dim=1)
            corrs = torch.stack(corrs, dim=1)
            sampled_future = torch.reshape(torch.stack(y, dim=1), (num_samples, -1, num_predicted_timesteps, pred_dim))

        # 

        y_dist = GMM2D(log_pis.reshape(num_samples, -1, num_predicted_timesteps, self.gmm_components),
                       mus.reshape(num_samples, -1, num_predicted_timesteps, self.gmm_components * self.pred_dim),
                       log_sigmas.reshape(num_samples, -1, num_predicted_timesteps, self.gmm_components * self.pred_dim),
                       corrs.reshape(num_samples, -1, num_predicted_timesteps, self.gmm_components),
                       self.log_sigma_min, self.log_sigma_max, self.device)

        if self.training:
            return y_dist
        else:
            return y_dist, sampled_future

    def decode(self ,x, y, z, TD, num_predicted_timesteps, num_samples, 
               sampling_decoding=True):
        y_dist = self.p_y_xz(x, z, TD, num_predicted_timesteps, num_samples, 
                             sampling_decoding=sampling_decoding)
        log_p_yt_xz = torch.clamp(y_dist.log_prob(y), max=self.log_p_yt_xz_max)
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)        
        return log_p_y_xz