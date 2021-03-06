import pickle
import numpy as np
import random
import torch
from tqdm import tqdm

from predictor.predictor import Predictor
from predictor.vanilla_lstm_predictor import VanillaLSTMPredictor
from data_utils.process_highd.track import *
import util
from args import args

PROCESSED_DATASET_PATH = './data/processed_data/highd/{}_uuid_to_track.pickle'
dt = 1 / DEFAULT_FRAME_RATE

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def infer_future_trajs(frame, sampled_vels, track):
    '''
        frame: initial frame ID
    '''
    sampled_future_trajs = []
    for i in range(len(sampled_vels)):
        x, y = track.xs[frame], track.ys[frame] # non-frenet
        traj_x, traj_y = [], []
        for t in range(args.n_pred_steps):
            x_dot, y_dot = sampled_vels[i][t]
            x_dot = denormalize(x_dot, MIN_ABS_VEL_X, MAX_ABS_VEL_X)
            y_dot = denormalize(y_dot, MIN_ABS_VEL_Y, MAX_ABS_VEL_Y)
            if track.driving_direction == 1:
                x = x - dt * x_dot
                y = y - dt * y_dot
            else:
                x = x + dt * x_dot
                y = y + dt * y_dot
            traj_x.append(x)
            traj_y.append(y)
        sampled_future_trajs.append(np.array([traj_x, traj_y]).T)

    return sampled_future_trajs

def predict_lane(frame, sampled_future_trajs, track, tol=0.1):
    '''
        Predict the lane ID of the endpoint of sampled_future_trajs
    '''
    x_error = []
    y_error = []
    dist_error = []
    votes = []
    init_lane_id = track.lane_ids[frame]
    lane_markings = list(track.upper_lane_markings) + list(track.lower_lane_markings)
    for i in range(len(sampled_future_trajs)):
        x_end, y_end = sampled_future_trajs[i][-1]
        _y_error = abs(y_end - track.ys[frame + args.n_pred_steps])
        _x_error = abs(x_end - track.xs[frame + args.n_pred_steps])
        _d_error = np.sqrt(_x_error ** 2 + _y_error ** 2)
        y_error.append(_y_error)
        x_error.append(_x_error)
        dist_error.append(_d_error)

        pred_lane_id = -1 # for outlier samples
        for k in range(len(lane_markings) - 1):
            upper = lane_markings[k]
            lower = lane_markings[k + 1]
            if upper < y_end < lower:
                if _y_error < tol:
                    # Handle lane marking border condition
                    pred_lane_id = track.lane_ids[frame + args.n_pred_steps]
                else:
                    pred_lane_id = k + 2
                break
        
        if pred_lane_id not in votes:
            votes.append(pred_lane_id)

    return votes, np.mean(x_error), np.mean(y_error), np.mean(dist_error)

def eval_lane_pred_accuracy(model, track, device):
    accuracy = [] 
    x_error = []
    y_error = []
    dist_error = []
    with tqdm(total=len(range(INPUT_SEQ_LEN, track.num_frames - PRED_SEQ_LEN))) as progress_bar:
        for frame in range(INPUT_SEQ_LEN, track.num_frames - PRED_SEQ_LEN):
            input_seq = track.state_tensors[frame - INPUT_SEQ_LEN : frame]
            input_masks = track.graph_masks[frame - INPUT_SEQ_LEN : frame]
            input_edge_types = track.edge_type_tensors[frame - INPUT_SEQ_LEN : frame]
    
            input_seq = torch.tensor(input_seq).float().to(device).unsqueeze(0)
            input_masks = torch.tensor(input_masks).float().to(device).unsqueeze(0)
            input_edge_types = torch.tensor(input_edge_types).float().to(device).unsqueeze(0)

            if args.model == 'cvae':
                sampled_future, z_p_samples = model.predict(
                    input_seq, input_masks, input_edge_types, args.n_z_samples_pred, most_likely=False)
                # sampled_future.shape = (n_z_samples_pred, 1, n_pred_steps, pred_dim), bs = 1
                sampled_vels = sampled_future.squeeze().detach().cpu().numpy() # (n_z_samples_pred, pred_seq_len, 2)
            elif args.model == 'vanilla':
                sampled_vels = model.predict(input_seq, args.n_pred_steps)
                sampled_vels = sampled_vels.detach().cpu().numpy()
                # print(sampled_vels.shape)
                # (1, n_pred_steps, 2), n_z_samples_pred = 1
    
            sampled_future_trajs = infer_future_trajs(frame, sampled_vels, track)
            true_future_lane_id = track.lane_ids[frame + PRED_SEQ_LEN]
            pred_future_lane_id_votes, pred_x_error, pred_y_error, pred_dist_error = \
                predict_lane(frame, sampled_future_trajs, track)
    
            # print(true_future_lane_id, pred_future_lane_id, pred_y_error)
    
            accuracy.append(true_future_lane_id in pred_future_lane_id_votes)
            x_error.append(pred_x_error)
            y_error.append(pred_y_error)
            dist_error.append(pred_dist_error)
    
            progress_bar.update(1)
            progress_bar.set_postfix(frame=frame,
                                     acc=sum(accuracy) / len(accuracy))

    accuracy = sum(accuracy) / len(accuracy) * 100
    y_error = np.mean(y_error)
    x_error = np.mean(x_error)
    dist_error = np.mean(dist_error)
    print('Lane ID prediction accuracy = {:.3f}%'.format(accuracy))
    print('Mean end-point prediction x error     = {:.3f} [meter]'.format(x_error))
    print('Mean end-point prediction y error     = {:.3f} [meter]'.format(y_error))
    print('Mean end-point prediction dist error     = {:.3f} [meter]'.format(dist_error))
    return accuracy, x_error, y_error, dist_error


mode = 'lane_pred'

def benchmarking(max_n_eval=10):
    dev_data_list=[14, 38, 23, 31, 20, 26, 32, 33, 17, 3, 27, 57, 49, 25, 55]
    # dev_data_list=[14]
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
    elif args.model == 'vanilla':
        model = VanillaLSTMPredictor(state_dim=args.state_dim,
                                     pred_dim=args.pred_dim,
                                     hidden_size=32,
                                     device=device)
    model = model.to(device)
    model.eval()

    model, step = util.load_model(model, args.load_path, args.gpu_ids)

    results = {
        'lane_pred_accuracy': [],
        'x_error': [],
        'y_error': [],
        'dist_error': [],
    }
    units = {
        'lane_pred_accuracy': '%',
        'x_error': '[meter]',
        'y_error': '[meter]',
        'dist_error': '[meter]',
    }
    # Prepare data
    for idx in range(len(dev_data_list)):
        data_id = dev_data_list[idx]
        data_str = '{:02d}'.format(data_id)
        with open(PROCESSED_DATASET_PATH.format(data_str), 'rb') as f:
            uuid_to_track = pickle.load(f)

        track_ids = list(range(1, len(uuid_to_track) + 1))
        random.shuffle(track_ids)

        n_eval = 0
        i = 0
        while n_eval < max_n_eval:
            track_id = track_ids[i]
            i += 1
            track = uuid_to_track[list(uuid_to_track.keys())[track_id - 1]]
            if track.num_lane_changes > 0 and track.num_frames > TOTAL_SEQ_LEN:
                # Only eval tracks with at least one lane change
                print(f'[{idx + 1}]/[{len(dev_data_list)}] : [{n_eval + 1}]/[{max_n_eval}]')
                print(f'Evaluating data_id = {data_str} track_id = {track_id}...')
                n_eval += 1
                track.generate_data_tensors(uuid_to_track)
                # Not converting to Frenet since we still want to know lane id info
                # track.convert_xy_to_frenet()
                lane_pred_accuracy, x_error, y_error, dist_error = eval_lane_pred_accuracy(model, track, device)
                results['lane_pred_accuracy'].append(lane_pred_accuracy)
                results['x_error'].append(x_error)
                results['y_error'].append(y_error)
                results['dist_error'].append(dist_error)

    print('=' * 80)
    print('Benchmarking results:')
    for k, v in results.items():
        print('{} = {:.3f} {}'.format(k, np.mean(v), units[k]))
    print('=' * 80)


if __name__ == '__main__':
    benchmarking(max_n_eval=10)