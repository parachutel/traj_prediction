import argparse
import torch

from data_utils.highd_dataset import build_highd_data_loader

from modules.encoders import Encoders
from modules.latent import DiscreteLatent


parser = argparse.ArgumentParser()
parser.add_argument('--test', type=str, default='encoders')
args = parser.parse_args()

device='cpu'

# Load data:
dataset_list = [1]
batch_size = 177
data_loader = build_highd_data_loader(dataset_list, batch_size, device=device)

encoders = Encoders(state_dim=6,
                    rel_state_dim=6,
                    edge_type_dim=4,
                    nhe_hidden_size=128,
                    ehe_hidden_size=128,
                    nfe_hidden_size=128,
                    device=device)

latent = DiscreteLatent(encoders.x_size + encoders.y_size,
                        encoders.x_size,
                        pred_dim=2, 
                        latent_dim=5,
                        kl_min=0.07, 
                        device=device)
# dummy settings for testing
latent.temp = 1.0
latent.z_logit_clip = 2.0

if args.test == 'encoders':
    print('Training')
    encoders.train()
    for input_seq, input_masks, input_edge_types, pred_seq in data_loader:
        x, y = encoders(input_seq, input_edge_types, pred_seq)
        print('x.shape =', x.shape)
        print('y.shape =', y.shape)
        break

    print('Eval')
    encoders.eval()
    for input_seq, input_masks, input_edge_types, pred_seq in data_loader:
        x = encoders(input_seq, input_edge_types, pred_seq)
        print('x.shape =', x.shape)
        break

elif args.test == 'latent':
    num_samples = 11
    print(args.test, 'Training')
    mode = 'training'
    for input_seq, input_masks, input_edge_types, pred_seq in data_loader:
        x, y = encoders(input_seq, input_edge_types, pred_seq)
        xy = torch.cat([x, y], dim=-1)
        print('x.shape =', x.shape)
        print('xy.shape =', xy.shape)
        
        # q(z | x, y)
        h_xy = latent.xy_to_latent(xy)
        latent.q_dist = latent.z_dist_from_hidden(h_xy, mode=mode)
        print('latent.q_dist =', latent.q_dist, latent.q_dist.probs.shape)
        z_q_samples = latent.sample_q(num_samples, mode=mode)
        print('z_q_samples.shape =', z_q_samples.shape)
        q_log_probs = latent.q_log_prob(z_q_samples)
        print('q_log_probs.shape =', q_log_probs.shape)
        
        # p(z | x)
        h_x = latent.x_to_latent(x)
        latent.p_dist = latent.z_dist_from_hidden(h_x, mode=mode)
        print('latent.p_dist =', latent.p_dist, latent.p_dist.probs.shape)
        z_p_samples = latent.sample_p(num_samples, mode=mode)
        print('z_p_samples.shape =', z_p_samples.shape)
        p_log_probs = latent.p_log_prob(z_p_samples)
        print('p_log_probs.shape =', p_log_probs.shape)
        
        kl = latent.kl_q_p()
        print('kl =', kl)
        break

    print(args.test, 'Eval')
    mode = 'eval'
    for input_seq, input_masks, input_edge_types, pred_seq in data_loader:
        x, y = encoders(input_seq, input_edge_types, pred_seq)
        xy = torch.cat([x, y], dim=-1)
        print('x.shape =', x.shape)
        print('xy.shape =', xy.shape)
        
        # q(z | x, y)
        h_xy = latent.xy_to_latent(xy)
        latent.q_dist = latent.z_dist_from_hidden(h_xy, mode=mode)
        print('latent.q_dist =', latent.q_dist, latent.q_dist.probs.shape)
        z_q_samples = latent.sample_q(num_samples, mode=mode)
        print('z_q_samples.shape =', z_q_samples.shape)
        q_log_probs = latent.q_log_prob(z_q_samples)
        print('q_log_probs.shape =', q_log_probs.shape)
        
        # p(z | x)
        h_x = latent.x_to_latent(x)
        latent.p_dist = latent.z_dist_from_hidden(h_x, mode=mode)
        print('latent.p_dist =', latent.p_dist, latent.p_dist.probs.shape)
        z_p_samples = latent.sample_p(num_samples, mode=mode)
        print('z_p_samples.shape =', z_p_samples.shape)
        p_log_probs = latent.p_log_prob(z_p_samples)
        print('p_log_probs.shape =', p_log_probs.shape)
        
        kl = latent.kl_q_p()
        print('kl =', kl)
        break


    print(args.test, 'Predict')
    mode = 'predict'
    for input_seq, input_masks, input_edge_types, pred_seq in data_loader:
        x = encoders(input_seq, input_edge_types, pred_seq)
        print('x.shape =', x.shape)
        
        # p(z | x)
        h_x = latent.x_to_latent(x)
        latent.p_dist = latent.z_dist_from_hidden(h_x, mode=mode)
        print('latent.p_dist =', latent.p_dist, latent.p_dist.probs.shape)
        z_p_samples = latent.sample_p(num_samples, mode=mode)
        print('z_p_samples.shape =', z_p_samples.shape)
        p_log_probs = latent.p_log_prob(z_p_samples)
        print('p_log_probs.shape =', p_log_probs.shape)
        
        kl = latent.kl_q_p()
        print('kl =', kl)
        break