import os
import sys
import pickle
import argparse
import uuid
import torch

from tqdm import tqdm
from data_management.read_csv import *
from track import Track
from matplotlib.animation import FFMpegFileWriter

import matplotlib.pyplot as plt

current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../../')

from args import args

RAW_DATASET_PATH = '../../data/raw_data/highd/data/'
PROCESSED_DATASET_PATH = '../../data/processed_data/highd/{}_uuid_to_track.pickle'
ANIMATION_PATH = '../../data/processed_data/highd/{}_{}_fps={}_animation.mp4'
TORCH_TENSOR_DIR = '../../data/processed_data/highd/input={}_pred={}_stride={}'

def load_highd_data(n_records):
    for data_id in tqdm(range(1, n_records + 1)):
        data_str = '{:02d}'.format(data_id)
        prefix = RAW_DATASET_PATH + data_str
        path_args = {}
        path_args['input_path'] = prefix + '_tracks.csv'
        path_args['input_static_path'] = prefix + '_tracksMeta.csv'
        path_args['input_meta_path'] = prefix + '_recordingMeta.csv'

        static_info = read_static_info(path_args) # Dict
        meta_dictionary = read_meta_info(path_args) # Dict
        tracks = read_track_csv(path_args) # List of Dict

        # Bounded to a specific data_id
        _tmp_uuid_to_track = {}
        _tmp_track_id_to_uuid = {track_id: str(uuid.uuid4()) for track_id in static_info}
        
        for t_id in range(1, len(static_info) + 1):
            track_obj = Track(data_id, tracks[t_id - 1], static_info[t_id], 
                              meta_dictionary, _tmp_track_id_to_uuid)
            _tmp_uuid_to_track[track_obj.uuid] = track_obj
        
        with open(PROCESSED_DATASET_PATH.format(data_str), 'wb') as f:
            pickle.dump(_tmp_uuid_to_track, f)

def plot_bbox(ax, x, y, w, h, color='k'):
    ax.plot([x - w / 2, x + w / 2], [y + h / 2, y + h / 2], color=color)
    ax.plot([x - w / 2, x - w / 2], [y - h / 2, y + h / 2], color=color)
    ax.plot([x + w / 2, x + w / 2], [y - h / 2, y + h / 2], color=color)
    ax.plot([x - w / 2, x + w / 2], [y - h / 2, y - h / 2], color=color)

def update_frame(i, ax, track, data_str, target_traj, neighbor_trajs, x_reversed, 
                 y_shift, initial_x, uuid_to_track):
    frame = track.frames[i]
    # Setup
    ax.set_xlim(0, 420)
    ax.set_ylim(-10, 10)
    ax.invert_yaxis()
    ax.set_title('Data = {}, Track ID = {}, Driving Direction = {}'.format(
                  data_str, int(track.track_id), int(track.driving_direction)))
    # ax.set_aspect('equal', 'box')
    plt.gcf().set_size_inches(20, 6)
    plt.tight_layout()
    ax.plot([0, 420], [0, 0], 
                 color='k', linewidth=1.5, linestyle='dashed')
    ax.plot([0, 420], [-track.lane_width / 2, -track.lane_width / 2], 
            color='k', linewidth=1.5)
    ax.plot([0, 420], [track.lane_width / 2, track.lane_width / 2], 
            color='k', linewidth=1.5)

    plt.arrow(x=10, y=3, dx=0, dy=4, color='k', width=0.5)
    plt.arrow(x=10, y=-3, dx=0, dy=-4, color='k', width=0.5)
    plt.annotate('left', (12, -7), fontsize=12)
    plt.annotate('right', (12, 7), fontsize=12)

    # Details
    ax.scatter(track.xs[i], track.ys[i], color='b')
    target_traj.append([track.xs[i], track.ys[i]])
    plot_bbox(ax, track.xs[i], track.ys[i], track.width, track.height, color='b')
        
    neighbors = track.neighbors[:, i]
    for n_uuid in neighbors:
        if n_uuid is not None:
            n_track = uuid_to_track[n_uuid]
            n_i = list(n_track.frames).index(frame)

            n_x = n_track.xs[n_i] - initial_x
            n_y = n_track.ys[n_i] - y_shift
            if x_reversed:
                n_x, n_y = -n_x, -n_y

            ax.scatter(n_x, n_y, color='r')
            plot_bbox(ax, n_x, n_y, n_track.width, n_track.height, color='r')
            ax.plot([n_x, track.xs[i]], [n_y, track.ys[i]], 
                        linestyle='dashed', color='g', linewidth=1)

            if n_uuid not in neighbor_trajs:
                neighbor_trajs[n_uuid] = []
            neighbor_trajs[n_uuid].append([n_x, n_y])

    # Trajectories
    _target_traj = np.array(target_traj)
    ax.plot(_target_traj[:, 0], _target_traj[:, 1], color='b', linewidth=1)
    # for n_uuid in neighbor_trajs:
    #     _n_traj = np.array(neighbor_trajs[n_uuid])
    #     ax.plot(_n_traj[:, 0], _n_traj[:, 1], color='r', linewidth=1)

def plot_one_tracks(data_id=1, track_id=1, fps=5):
    data_str = '{:02d}'.format(data_id)
    with open(PROCESSED_DATASET_PATH.format(data_str), 'rb') as f:
        uuid_to_track = pickle.load(f)

    track = uuid_to_track[list(uuid_to_track.keys())[track_id - 1]]
    
    track.generate_data_tensors(uuid_to_track)

    x_reversed, y_shift, initial_x = track.convert_xy_to_frenet()
    assert track.track_id == track_id

    fig, ax = plt.subplots()
    fig.set_figwidth(25) # inches

    target_traj = []
    neighbor_trajs = {}

    moviewriter = FFMpegFileWriter(fps=fps) # track.frame_rate
    print('INFO: Generating animation for Data {}, Track {}'.format(data_str, track_id))
    with moviewriter.saving(fig, ANIMATION_PATH.format(data_str, track_id, fps), dpi=300):
        for i in tqdm(range(len(track.frames))):
            update_frame(i, ax, track, data_str, target_traj, neighbor_trajs, 
                         x_reversed, y_shift, initial_x, uuid_to_track)
            moviewriter.grab_frame()
            plt.draw()
            plt.pause(0.01)
            ax.cla()


# Plot absolute lane marks
# for i, y in enumerate(track.upper_lane_markings):
#     if 0 < i < len(track.upper_lane_markings) - 1:
#         linestyle = 'dashed'
#     else:
#         linestyle = None    
#     plt.plot([0, 400], [y, y], color='k', linewidth=3, linestyle=linestyle)

# for i, y in enumerate(track.lower_lane_markings):
#     if 0 < i < len(track.lower_lane_markings) - 1:
#         linestyle = 'dashed'
#     else:
#         linestyle = None    
#     plt.plot([0, 400], [y, y], color='k', linewidth=3, linestyle=linestyle)


def generate_masks_and_state_tensors(data_id=1, track_id=1):
    print('Testing generate_data_tensors')
    data_str = '{:02d}'.format(data_id)
    with open(PROCESSED_DATASET_PATH.format(data_str), 'rb') as f:
        uuid_to_track = pickle.load(f)

    track = uuid_to_track[list(uuid_to_track.keys())[track_id - 1]]
    track.generate_data_tensors(uuid_to_track)

    print('track.graph_masks.shape =', track.graph_masks.shape)
    print('track.state_tensors.shape =', track.state_tensors.shape)

    # state = [x, y, vel_x, vel_y, acc_x, acc_y]
    bins = np.linspace(-1, 1, 50)
    plt.hist(track.state_tensors[:, 1, 1, 0].flatten(), bins=bins, alpha=0.7, label='self.x')
    plt.hist(track.state_tensors[:, 1, 1, 1].flatten(), bins=bins, alpha=0.7, label='self.y')
    plt.hist(track.state_tensors[:, 1, 1, 2].flatten(), bins=bins, alpha=0.7, label='self.vel_x')
    plt.hist(track.state_tensors[:, 1, 1, 3].flatten(), bins=bins, alpha=0.7, label='self.vel_y')
    plt.hist(track.state_tensors[:, 1, 1, 4].flatten(), bins=bins, alpha=0.7, label='self.acc_x')
    plt.hist(track.state_tensors[:, 1, 1, 5].flatten(), bins=bins, alpha=0.7, label='self.acc_y')

    plt.title(f'Dataset {args.dataset_id}, Track {args.track_id}, normalized self state entry distribution')
    plt.legend()
    plt.show()

def generate_training_data_segments(dataset_list=None):

    if not dataset_list:
        dataset_list = range(1, args.n_records + 1) # all
    assert max(dataset_list) <= args.n_records

    for k, data_id in enumerate(dataset_list):
        data_str = '{:02d}'.format(data_id)
        print(f'INFO: processing {data_str} ({k + 1}/{len(dataset_list)})')
        with open(PROCESSED_DATASET_PATH.format(data_str), 'rb') as f:
            uuid_to_track = pickle.load(f)

        input_seq_segments = []
        input_masks_segments = []
        input_edge_types_segments = []
        pred_seq_seqments = []
        uuids = list(uuid_to_track.keys())
        for i in tqdm(range(len(uuid_to_track))):
            uuid = uuids[i]
            track = uuid_to_track[uuid]
            track.generate_data_tensors(uuid_to_track)
            track_data_segments = track.generate_training_data_segments()

            if track_data_segments:
                for input_seq, input_masks, input_edge_types, pred_seq in track_data_segments:
                    input_seq_segments.append(input_seq)
                    input_masks_segments.append(input_masks)
                    input_edge_types_segments.append(input_edge_types)
                    pred_seq_seqments.append(pred_seq)

        input_seq_segments = torch.tensor(input_seq_segments).float()
        input_masks_segments = torch.tensor(input_masks_segments).float()
        input_edge_types_segments = torch.tensor(input_edge_types_segments).float()
        pred_seq_seqments = torch.tensor(pred_seq_seqments).float()

        torch_tensor_dir = TORCH_TENSOR_DIR.format(
            args.input_seconds, args.pred_seconds, args.forward_shift_seconds)
        if not os.path.exists(torch_tensor_dir):
            os.makedirs(torch_tensor_dir)

        torch_tensor_path = torch_tensor_dir + '/{}_{}.pt'
        torch.save(input_seq_segments, torch_tensor_path.format(data_str, 'input_seq'))
        torch.save(input_masks_segments, torch_tensor_path.format(data_str, 'input_masks'))
        torch.save(input_edge_types_segments, torch_tensor_path.format(data_str, 'input_edge_types'))
        torch.save(pred_seq_seqments, torch_tensor_path.format(data_str, 'pred_seq'))

if __name__ == '__main__':
    
    if args.mode == 'pickle':
        load_highd_data(args.n_records)
    elif args.mode == 'animation':
        plot_one_tracks(args.dataset_id, args.track_id, args.fps)
    elif args.mode == 'test_states':
        generate_masks_and_state_tensors(args.dataset_id, args.track_id)
    elif args.mode == 'data_segmentation':
        # 1-15, 16-30, 31-45, 46-60
        generate_training_data_segments(range(args.start, args.end + 1))