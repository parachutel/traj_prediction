import numpy as np

class PlusAITrack:
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
        '''
            Failing
        '''
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
    def duration(self): # in seconds
        if self.time_sorted:
            return self.time_stamps[-1] - self.time_stamps[0]
        else:
            return max(self.time_stamps) - min(self.time_stamps)

    @property
    def total_displacement(self):
        return np.linalg.norm([self.xs[-1] - self.xs[0], self.ys[-1] - self.ys[0]])