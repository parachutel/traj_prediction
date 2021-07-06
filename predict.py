import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
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
from data_utils.process_highd.track import *

import util
from args import args

def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args, training=False)
    log = util.get_logger(args.save_dir, args.name)
    device, args.gpu_ids = util.get_available_devices()
    # device, args.gpu_ids = 'cpu', None
    log.info(f'Using device {device}...')
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
    dataset_size = len(test_loader.dataset)
    log.info(f'Test dataset size = {dataset_size}')

    # Predict
    log.info('Predicting...')
    vis_idxs = np.random.randint(0, dataset_size, args.n_vis)

    vis_data = test_loader.dataset[vis_idxs]
    input_seq, _, input_edge_types, pred_seq = vis_data

    sampled_future, z_p_samples = model.predict(
        input_seq, input_edge_types, args.n_z_samples_pred, most_likely=False)
    # sampled_future.shape = (n_z_samples, n_vis, pred_seq_len, 2)

    sampled_future = sampled_future.detach().cpu().numpy()
    input_seq = input_seq.detach().cpu().numpy()
    pred_seq = pred_seq.detach().cpu().numpy()

    for traj_id in tqdm(range(args.n_vis)):
        sampled_vels = sampled_future[:, traj_id] # (x_dot, y_dot), (n_samples, pred_seq_len, 2)
        input_traj = input_seq[traj_id, :, 1, 1, :2] # (x, y), (in_seq_len, 2)
        pred_traj = pred_seq[traj_id, :, 1, 1, :2] # (x, y), (pred_seq_len, 2)
        
        plt.plot(denormalize(input_traj[:, 0], MIN_X, MAX_X), input_traj[:, 1] * LANE_WIDTH, color='blue')
        plt.plot(denormalize(pred_traj[:, 0], MIN_X, MAX_X), pred_traj[:, 1] * LANE_WIDTH, color='green')
        
        ground_truth_vels = pred_seq[traj_id, :, 1, 1, 2:4]
        
        dt = 1/25
        x_t, y_t = denormalize(input_traj[-1, 0], MIN_X, MAX_X), input_traj[-1, 1] * LANE_WIDTH
        traj_x_t, traj_y_t = [], []
        for t in range(args.n_pred_steps):
            x_dot_t, y_dot_t = ground_truth_vels[t]
            x_dot_t = denormalize(x_dot_t, MIN_ABS_VEL_X, MAX_ABS_VEL_X)
            y_dot_t = denormalize(y_dot_t, MIN_ABS_VEL_Y, MAX_ABS_VEL_Y)
            x_t = x_t + dt * x_dot_t
            y_t = y_t + dt * y_dot_t
            traj_x_t.append(x_t)
            traj_y_t.append(y_t)
        plt.plot(traj_x_t, traj_y_t, color='m')
        for i in range(args.n_z_samples_pred):
            x, y = denormalize(input_traj[-1, 0], MIN_X, MAX_X), input_traj[-1, 1] * LANE_WIDTH
            _sampled_vels = sampled_vels[i] # (pred_seq_len, 2)
            traj_x, traj_y = [], []
            for t in range(args.n_pred_steps):
                x_dot, y_dot = _sampled_vels[t]
                x_dot = denormalize(x_dot, MIN_ABS_VEL_X, MAX_ABS_VEL_X)
                y_dot = denormalize(y_dot, MIN_ABS_VEL_Y, MAX_ABS_VEL_Y)
                x = x + dt * x_dot
                y = y + dt * y_dot
                traj_x.append(x)
                traj_y.append(y)
            plt.plot(traj_x, traj_y, color='red', alpha=0.5)

        plt.legend(['Input Traj.', 'True Future Traj.', 'True Vel-Inferred Future Traj.', 'Predictions'])
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        # plt.gcf().set_figwidth(25) # inches
        # plt.gca().set_aspect('equal', 'box')
        # plt.gcf().set_size_inches(20, 6)
        # plt.gca().set_xlim(0, 420)
        # plt.gca().set_ylim(-6, 6)
        plt.tight_layout()
        plt.savefig(args.save_dir + f'/{traj_id}.png')
        plt.gca().cla()

if __name__ == '__main__':
    main(args)
