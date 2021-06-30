import os

import ujson as json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import interpolate


# dataset_name = 'batch_20190614T164101'
dataset_name = 'batch_20200108T141022_changzhou'
dataset_parent_path = '../data/raw_data/plusai/train/'

dataset_path = os.path.join(dataset_parent_path, dataset_name)
summary_json_path = os.path.join(dataset_path, dataset_name + '.json')


class CarTrajectory:
    def __init__(self, uuid, car_type):
        self.uuid = uuid
        self.type = car_type
        self.time_stamps = []
        self.xs = []
        self.ys = []
        self.yaws = []
        self.vel_xs = []
        self.vel_ys = []
        self.acc_xs = []
        self.acc_ys = []

        self.ego_xs = []
        self.ego_ys = []
        self.ego_headings = []
        self.ego_vel_xs = []
        self.ego_vel_ys = []
        self.ego_acc_xs = []
        self.ego_acc_ys = []

        self.neighbors = [] # <--hard to get

        self.time_sorted = False

    def add_record(self, t, x, y, yaw, ego_x, ego_y):
        self.time_stamps.append(t)
        self.xs.append(x)
        self.ys.append(y)
        self.yaws.append(yaw)
        self.ego_xs.append(ego_x)
        self.ego_ys.append(ego_y)
        # self.ego_yaw.append(ego_yaw)

    def smooth_obstacle_data(self):
        x, y = np.array(self.xs), np.array(self.ys)

        points = np.vstack((x, y)).T
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]
        splines = [interpolate.UnivariateSpline(distance, coords, k=3, s=.2) for coords in points.T]
        alpha = np.linspace(0, 1, len(x))
        points_fitted = np.vstack([spl(alpha) for spl in splines]).T
        self.xs = points_fitted[:, 0].T
        self.ys = points_fitted[:, 1].T

    def convert_coord_system(self):
        pass

    def infer_kinematics(self):
        pass

    def sort_by_time(self):
        self.time_sorted = True
        zipped = zip(self.time_stamps, self.xs, self.ys, self.yaws, self.ego_xs, self.ego_ys)
        zipped = sorted(zipped)
        for i, items in enumerate(zipped):
            self.time_stamps[i] = items[0]
            self.xs[i] = items[1]
            self.ys[i] = items[2]
            self.yaws[i] = items[3]
            self.ego_xs[i] = items[4]
            self.ego_ys[i] = items[5]

    @property
    def n_frames(self):
        return len(self.time_stamps)


    @property
    def duration(self):
        if self.time_sorted:
            return self.time_stamps[-1] - self.time_stamps[0]
        else:
            return max(self.time_stamps) - min(self.time_stamps)


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
                obstacle_car_trajectories[uuid] = CarTrajectory(uuid, car_type)
            x = car['bottom_center']['x']
            y = car['bottom_center']['y']
            yaw = car['yaw']
            obstacle_car_trajectories[uuid].add_record(t, x, y, yaw, ego_x, ego_y)

    for uuid in obstacle_car_trajectories:
        obstacle_car_trajectories[uuid].sort_by_time()

    print(f'INFO: {len(obstacle_car_trajectories)} CarTrajectory was found.')

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
        if traj.duration > 7:
            # Skipping short trajectories
            traj.smooth_obstacle_data()
            # dxs = np.array(traj.xs) - np.array(traj.ego_xs)
            # dys = np.array(traj.ys) - np.array(traj.ego_ys)
            # plt.plot(dxs, dys)
            plt.plot(traj.xs, traj.ys)
            # plt.plot(traj.ego_xs, traj.ego_ys, color='red', linewidth=2)
    plt.show()