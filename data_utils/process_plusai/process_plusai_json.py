import os

import ujson as json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from plusai_track import PlusAITrack

# dataset_name = 'batch_20191101T131603'
# dataset_name = 'batch_20191108T143338'
# dataset_name = 'batch_20190614T164101'
dataset_name = 'batch_20200108T141022_changzhou'
dataset_parent_path = '../../data/raw_data/plusai/train/'


dataset_path = os.path.join(dataset_parent_path, dataset_name)
summary_json_path = os.path.join(dataset_path, dataset_name + '.json')



def load_json(json_path):
    with open(json_path) as jf:
        data = json.load(jf)
    return data


def process_summary_json(summary_json_path):
    data = load_json(summary_json_path)
    frame_labelings = sorted(data['labeling'], key=lambda x: x['filename'])
    print(f'INFO: {len(frame_labelings)} frames in {dataset_name}.json.')

    obstacle_car_trajectories = {}
    for frame in frame_labelings:

        frame_json_path = os.path.join(dataset_parent_path, frame['filename'])
        frame_meta_data = load_json(frame_json_path)
        t = frame_meta_data['timestamp']
        ego_x = frame_meta_data['car_position']['x']
        ego_y = frame_meta_data['car_position']['y']
        # ego_yaw = frame_meta_data[]

        for car in frame['annotations_3d']:
            uuid = car['uuid']
            if uuid not in obstacle_car_trajectories:
                car_type = car['attribute']['label_type']
                obstacle_car_trajectories[uuid] = PlusAITrack(uuid, car_type)
            x = car['bottom_center']['x']
            y = car['bottom_center']['y']
            yaw = car['yaw']
            obstacle_car_trajectories[uuid].add_record(t, x, y, yaw, ego_x, ego_y)

    for uuid in obstacle_car_trajectories:
        obstacle_car_trajectories[uuid].sort_by_time()

    print(f'INFO: {len(obstacle_car_trajectories)} trajectories was found.')

    return obstacle_car_trajectories

def traj_stats(trajectories):
    durations = {(t, t + 1): 0 for t in range(10)}
    durations['>= 10'] = 0
    tot = len(trajectories)
    for uuid in trajectories:
        t = int(trajectories[uuid].duration)
        if t < 10:
            durations[(t, t + 1)] += 1
        else:
            durations['>= 10'] += 1

    for key in durations:
        durations[key] /= (tot / 100)
        print('STATS: {} sec: {:.2f}%'.format(key, durations[key]))


def infer_neighbors(trajectories):
    pass

if __name__ == '__main__':
    obstacle_car_trajectories = process_summary_json(summary_json_path)
    traj_stats(obstacle_car_trajectories)
    for uuid in obstacle_car_trajectories:
        traj = obstacle_car_trajectories[uuid]
        print(uuid, traj.total_displacement)
        if traj.total_displacement > 50:
            # Skipping short trajectories
            # traj.smooth_obstacle_data()
            # dxs = np.array(traj.xs) - np.array(traj.ego_xs)
            # dys = np.array(traj.ys) - np.array(traj.ego_ys)
            # plt.plot(dxs, dys)
            plt.plot(traj.xs, traj.ys)
            plt.plot(traj.ego_xs, traj.ego_ys, color='k')
            # plt.plot(traj.ego_xs, traj.ego_ys, color='red', linewidth=2)
            plt.show()