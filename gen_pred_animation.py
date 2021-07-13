import pickle
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegFileWriter

from predictor.predictor import Predictor
from predictor.vanilla_lstm_predictor import VanillaLSTMPredictor
from data_utils.process_highd.track import *
from data_utils.process_highd.process_highd import *
import util
from args import args

PROCESSED_DATASET_PATH = './data/processed_data/highd/{}_uuid_to_track.pickle'
ANIMATION_DIR = './save/animation/highd_input={}_pred={}_stride={}'.format(
    args.input_seconds, args.pred_seconds, args.forward_shift_seconds)
ANIMATION_PATH = ANIMATION_DIR + '/{}_{}_{}_fps={}_animation.mp4'

dt = 1 / DEFAULT_FRAME_RATE

if not os.path.exists(ANIMATION_DIR):
    os.makedirs(ANIMATION_DIR)

def plot_sampled_future(frame, ax, sampled_vels, track):
    for i in range(len(sampled_vels)):
        x, y = track.xs[frame], track.ys[frame]
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
        ax.plot(traj_x, traj_y, color='red', alpha=0.5)

def gen_pred_animation(data_id=1, track_id=1, fps=20):
    global ANIMATION_PATH
    # Load model
    device, args.gpu_ids = util.get_available_devices()
    if args.model == 'cvae':
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
        args.name = args.model + ('_zbest' if args.most_likely else '_full')
    elif args.model == 'vanilla':
        model = VanillaLSTMPredictor(state_dim=args.state_dim,
                                     pred_dim=args.pred_dim,
                                     hidden_size=32,
                                     device=device)
        args.name = args.model
    model = model.to(device)
    model.eval()

    print(f'Loading checkpoint from {args.load_path}...')
    model, step = util.load_model(model, args.load_path, args.gpu_ids)

    # Prepare data
    data_str = '{:02d}'.format(data_id)
    ANIMATION_PATH = ANIMATION_PATH.format(args.name, data_str, track_id, fps)
    with open(PROCESSED_DATASET_PATH.format(data_str), 'rb') as f:
        uuid_to_track = pickle.load(f)

    track = uuid_to_track[list(uuid_to_track.keys())[track_id - 1]]
    track.generate_data_tensors(uuid_to_track)
    # (seq_len, 3, 3, state_dim)
    x_reversed, y_shift, initial_x = track.convert_xy_to_frenet()

    # Prepare plot
    fig, ax = plt.subplots()
    # fig.set_figwidth(25) # inches
    moviewriter = FFMpegFileWriter(fps=fps)

    target_traj = []
    neighbor_trajs = {}

    # Append the first INPUT_SEQ_LEN frames if target traj
    for i in range(INPUT_SEQ_LEN):
        target_traj.append([track.xs[i], track.ys[i]])

    with moviewriter.saving(fig, ANIMATION_PATH, dpi=300):
        for frame in tqdm(range(INPUT_SEQ_LEN, track.num_frames)):
            input_seq = track.state_tensors[frame - INPUT_SEQ_LEN : frame]
            input_masks = track.graph_masks[frame - INPUT_SEQ_LEN : frame]
            input_edge_types = track.edge_type_tensors[frame - INPUT_SEQ_LEN : frame]
            assert len(input_seq) == INPUT_SEQ_LEN
    
            input_seq = torch.tensor(input_seq).float().to(device).unsqueeze(0)
            input_edge_types = torch.tensor(input_edge_types).float().to(device).unsqueeze(0)
            
            if args.model == 'cvae':
                sampled_future, z_p_samples = model.predict(
                    input_seq, input_masks, input_edge_types, args.n_z_samples_pred, most_likely=False)
                # sampled_future.shape = (n_z_samples_pred, 1, n_pred_steps, pred_dim), bs = 1
            elif args.model == 'vanilla':
                sampled_future = model.predict(input_seq, args.n_pred_steps)
                sampled_future = sampled_future.unsqueeze(0)
                # (1, 1, n_pred_steps, pred_dim), n_z_samples_pred = 1
            sampled_vels = sampled_future.squeeze(1).detach().cpu().numpy() # (n_z_samples_pred, pred_seq_len, 2)

            # Plot sampled_future
            plot_sampled_future(frame, ax, sampled_vels, track)

            # Plot other things in the frame
            update_frame(frame, ax, track, data_str, target_traj, neighbor_trajs, x_reversed, y_shift, 
                         initial_x, uuid_to_track)
            moviewriter.grab_frame()
            plt.draw()
            # plt.show()
            # plt.pause(0.01)
            ax.cla()

if __name__ == '__main__':
    gen_pred_animation(args.dataset_id, args.track_id)