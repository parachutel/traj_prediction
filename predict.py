import math
import numpy as np
import matplotlib.pyplot as plt
import random
from json import dumps
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched

from predictor.predictor import Predictor

from data_utils.highd_dataset import build_highd_data_loader
import util
from args import args

def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args, training=False)
    log = util.get_logger(args.save_dir, args.name)
    # tbx = SummaryWriter(args.save_dir)
    # device, args.gpu_ids = util.get_available_devices()
    device, args.gpu_ids = 'cpu', None
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    log.info('Building model...')
    model = Predictor(state_dim=args.state_dim,
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
                      device=device)
    
    log.info(f'Loading checkpoint from {args.load_path}...')
    model, step = util.load_model(model, args.load_path, args.gpu_ids)

    model = model.to(device)
    model.eval()

    log.info('Building dataset...')
    test_data_list = [1]
    test_loader = build_highd_data_loader(
        test_data_list, args.eval_batch_size, device=device)

    # Predict
    log.info('Predicting...')
    num_pred_samples = 10
    with torch.no_grad(), tqdm(total=len(test_loader.dataset)) as progress_bar:
        for input_seq, _, input_edge_types, pred_seq in test_loader:
            sampled_future, z_p_samples = model.predict(
                input_seq, input_edge_types, num_pred_samples, most_likely=False)

            
            sampled_future = sampled_future.detach().cpu().numpy()
            input_seq = input_seq.detach().cpu().numpy()
            pred_seq = pred_seq.detach().cpu().numpy()

            traj_id = 1

            sampled_trajs = sampled_future[:, traj_id] # (x_dot, y_dot), (n_samples, pred_seq_len, 2)
            input_traj = input_seq[traj_id, :, 1, 1, :2] # (x, y), (in_seq_len, 2)
            ground_truth = pred_seq[traj_id, :, 1, 1, :2] # (x, y), (pred_seq_len, 2)

            plt.plot(input_traj[:, 0], input_traj[:, 1])
            plt.plot(ground_truth[:, 0], ground_truth[:, 1])

            
            # mean_sampled_vels = np.mean(sampled_trajs, axis=0)
            ground_truth_vels = pred_seq[traj_id, :, 1, 1, 2:4]
            # print(np.mean(mean_sampled_vels - ground_truth_vels))
            dt = 1/100

            x_t, y_t = input_traj[-1, :]
            traj_x_t, traj_y_t = [], []
            for t in range(args.n_pred_steps):
                x_dot_t, y_dot_t = ground_truth_vels[t]
                x_t = x_t - dt * x_dot_t
                y_t = y_t + dt * y_dot_t
                traj_x_t.append(x_t)
                traj_y_t.append(y_t)
            plt.plot(traj_x_t, traj_y_t, color='green')

            for i in range(num_pred_samples):
                x, y = input_traj[-1, :]
                sampled_traj = sampled_trajs[i] # (pred_seq_len, 2)
                traj_x, traj_y = [], []
                for t in range(args.n_pred_steps):
                    x_dot, y_dot = sampled_traj[t]
                    x = x - dt * x_dot
                    y = y + dt * y_dot
                    traj_x.append(x)
                    traj_y.append(y)
                plt.plot(traj_x, traj_y, color='red')
            plt.show()

            break

if __name__ == '__main__':
    main(args)
