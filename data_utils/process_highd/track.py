from data_management.read_csv import *
import numpy as np
import torch

DEFAULT_FRAME_RATE = 25 # hz
FORWARD_SHIFT_STRIDE = DEFAULT_FRAME_RATE * 2
INPUT_SEQ_LEN = DEFAULT_FRAME_RATE * 4
PRED_SEQ_LEN = DEFAULT_FRAME_RATE * 2
TOTAL_SEQ_LEN = INPUT_SEQ_LEN + PRED_SEQ_LEN
MIN_NUM_FRAMES = TOTAL_SEQ_LEN

# Approximate emprical bounds from highD for normalizing data to (0, 1)
MIN_X = 0
MAX_X = 420
# y is normalized by lane_width / 2
MIN_ABS_VEL_X = 30
MAX_ABS_VEL_X = 50
MIN_ABS_VEL_Y = 0
MAX_ABS_VEL_Y = 1.2
MIN_ABS_ACC_X = 0
MAX_ABS_ACC_X = 2
MIN_ABS_ACC_Y = 0
MAX_ABS_ACC_Y = 0.5

INTERACTION_TYPE_TO_ONEHOT = {
    'Car-Car':     np.array([1, 0, 0 ,0]),
    'Car-Truck':   np.array([0, 1, 0 ,0]),
    'Truck-Car':   np.array([0, 0, 1 ,0]),
    'Truck-Truck': np.array([0, 0, 0 ,1]),
}

NEIGHBOR_IDX_TO_COORDS = {
    #    0    1    2
    # 0 [4]  [0]  [7]
    # 1 [3]   X   [6]
    # 2 [2]  [1]  [5]
    
    0: (0, 1),
    1: (2, 1),
    2: (2, 0),
    3: (1, 0),
    4: (0, 0),
    5: (2, 2),
    6: (1, 2),
    7: (0, 2),
}

def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


class Track:
    def __init__(self, data_id, track, static_info, meta_dictionary, track_id_to_uuid):
        self.data_id = data_id
        self.set_info(track, static_info, meta_dictionary, track_id_to_uuid)


    def set_info(self, track, static_info, meta_dictionary, track_id_to_uuid):
        # static
        self.track_id = static_info[TRACK_ID]
        self.uuid = track_id_to_uuid[static_info[TRACK_ID]]
        self.initial_frame = static_info[INITIAL_FRAME]
        self.final_frame = static_info[FINAL_FRAME]
        self.num_frames = static_info[NUM_FRAMES]
        self.v_type = static_info[CLASS]
        self.driving_direction = static_info[DRIVING_DIRECTION]
        self.num_lane_changes = static_info[NUMBER_LANE_CHANGES]

        # meta
        self.frame_rate = meta_dictionary[FRAME_RATE]
        self.upper_lane_markings = meta_dictionary[UPPER_LANE_MARKINGS]
        self.lower_lane_markings = meta_dictionary[LOWER_LANE_MARKINGS]

        # track
        self.frames = track[FRAME]
        ## Convert upper left corner x y to upper center x y
        self.width = track[BBOX][:, 2]
        self.height = track[BBOX][:, 3]
        self.xs = track[BBOX][:, 0] + self.width / 2 # x + w / 2 
        self.ys = track[BBOX][:, 1] + self.height / 2 # y + h / 2
        self.vel_xs = track[X_VELOCITY]
        self.vel_ys = track[Y_VELOCITY]
        self.acc_xs = track[X_ACCELERATION]
        self.acc_ys = track[Y_ACCELERATION]
        self.front_dists = track[FRONT_SIGHT_DISTANCE]
        self.back_dists = track[BACK_SIGHT_DISTANCE]
        self.thws = track[THW]
        self.ttcs = track[TTC]
        self.dhws = track[DHW]
        self.preceding_vel_xs = track[PRECEDING_X_VELOCITY]
        self.neighbor_preceding_uuids = [track_id_to_uuid[t_id] if t_id != 0 else None 
                                         for t_id in track[PRECEDING_ID]]
        self.neighbor_following_uuids = [track_id_to_uuid[t_id] if t_id != 0 else None 
                                         for t_id in track[FOLLOWING_ID]]
        self.neighbor_left_following_uuids = [track_id_to_uuid[t_id] if t_id != 0 else None 
                                              for t_id in track[LEFT_FOLLOWING_ID]]
        self.neighbor_left_alongside_uuids = [track_id_to_uuid[t_id] if t_id != 0 else None 
                                              for t_id in track[LEFT_ALONGSIDE_ID]]
        self.neighbor_left_preceding_uuids = [track_id_to_uuid[t_id] if t_id != 0 else None 
                                              for t_id in track[LEFT_PRECEDING_ID]]
        self.neighbor_right_following_uuids = [track_id_to_uuid[t_id] if t_id != 0 else None 
                                               for t_id in track[RIGHT_FOLLOWING_ID]]
        self.neighbor_right_alongside_uuids = [track_id_to_uuid[t_id] if t_id != 0 else None 
                                               for t_id in track[RIGHT_ALONGSIDE_ID]]
        self.neighbor_right_preceding_uuids = [track_id_to_uuid[t_id] if t_id != 0 else None 
                                              for t_id in track[RIGHT_PRECEDING_ID]]
        self.lane_ids = track[LANE_ID]

        # custom
        self.get_y_lane_center()
        self.frenet = False
        self.frenet_data = self.get_frenet_data()
        self.x_reversed = False
        
        # States
        self.state_len = len(self.get_state(0))


    def get_y_lane_center(self):
        lane_id = self.lane_ids[0]
        n_upper_lane_markings = len(self.upper_lane_markings)
        if self.driving_direction == 1:
            y_lane_center = np.mean([self.upper_lane_markings[lane_id - 2],
                                     self.upper_lane_markings[lane_id - 1]])
            self.lane_width = self.upper_lane_markings[lane_id - 1] \
                - self.upper_lane_markings[lane_id - 2]
        else:
            y_lane_center = np.mean(
                [self.lower_lane_markings[lane_id - n_upper_lane_markings - 2],
                 self.lower_lane_markings[lane_id - n_upper_lane_markings - 1]])
            self.lane_width = self.lower_lane_markings[lane_id - n_upper_lane_markings - 1] \
                - self.lower_lane_markings[lane_id - n_upper_lane_markings - 2]

        self.y_lane_center = y_lane_center


    def convert_xy_to_frenet(self):
        '''
            Modifies self attributes 
            For visualization only
        '''
        self.frenet = True
        initial_x = self.xs[0]
        self.ys -= self.y_lane_center
        if self.driving_direction == 1:
            # Reverse the x-arraies:
            self.x_reversed = True
            self.xs = initial_x - self.xs
            self.ys = -self.ys

        return self.x_reversed, self.y_lane_center, initial_x

    def get_frenet_data(self):
        '''
            NOT modifies self attributes
        '''
        initial_x = self.xs[0]
        xs = -initial_x + self.xs
        vel_xs = self.vel_xs
        acc_xs = self.acc_xs
        ys = -self.y_lane_center + self.ys
        vel_ys = self.vel_ys
        acc_ys = self.acc_ys

        if self.driving_direction == 1:
            return (-xs, -ys, -vel_xs, -vel_ys, -acc_xs, -acc_ys)
        
        return (xs, ys, vel_xs, vel_ys, acc_xs, acc_ys)

    @property
    def duration(self):
        return self.num_frames / self.frame_rate


    @property
    def neighbors(self):
        # shape = (8, num_frames)
        #    0    1    2
        # 0 [4]  [0]  [7]
        # 1 [3]   X   [6]
        # 2 [2]  [1]  [5]
        return np.vstack((self.neighbor_preceding_uuids, self.neighbor_following_uuids,
            self.neighbor_left_following_uuids, self.neighbor_left_alongside_uuids,
            self.neighbor_left_preceding_uuids, self.neighbor_right_following_uuids,
            self.neighbor_right_alongside_uuids, self.neighbor_right_preceding_uuids))


    def generate_local_interaction_temporal_graph_masks(self):
        self.graph_masks = np.zeros((self.num_frames, 3, 3))

        for t in range(self.num_frames):
            neighbors = self.neighbors[:, t]
            for i, neighbor in enumerate(neighbors):
                if neighbor is not None:
                    coords = (t,) + NEIGHBOR_IDX_TO_COORDS[i]
                    self.graph_masks[coords] = 1

        return self.graph_masks


    def get_state(self, i):
        # In Frenet system
        # i = frame_idx
        xs, ys, vel_xs, vel_ys, acc_xs, acc_ys = self.frenet_data
        state = [xs[i], ys[i], vel_xs[i], vel_ys[i], acc_xs[i], acc_ys[i]]
        return np.array(state)


    def get_relative_state(self, frame_idx, ref_frame_idx, ref_track):
        # Relative values in non-Frenet system
        rel_x = self.xs[frame_idx] - ref_track.xs[ref_frame_idx]
        rel_vel_x = self.vel_xs[frame_idx] - ref_track.vel_xs[ref_frame_idx]
        rel_acc_x = self.acc_xs[frame_idx] - ref_track.acc_xs[ref_frame_idx]
        rel_y = self.ys[frame_idx] - ref_track.ys[ref_frame_idx]
        rel_vel_y = self.vel_ys[frame_idx] - ref_track.vel_ys[ref_frame_idx]
        rel_acc_y = self.acc_ys[frame_idx] - ref_track.acc_ys[ref_frame_idx]

        if self.driving_direction == 1:
            return np.array([-rel_x, -rel_y, -rel_vel_x, -rel_vel_y, -rel_acc_x, -rel_acc_y])

        return np.array([rel_x, rel_y, rel_vel_x, rel_vel_y, rel_acc_x, rel_acc_y])


    def normalize_state(self, state):
        # state is in Frenet system
        # state = [x, y, vel_x, vel_y, acc_x, acc_y]
        x, y, vel_x, vel_y, acc_x, acc_y = tuple(state)
        x = normalize(x, MIN_X, MAX_X)
        y /= self.lane_width
        vel_x = normalize(vel_x, MIN_ABS_VEL_X, MAX_ABS_VEL_X)
        vel_y = normalize(vel_y, MIN_ABS_VEL_Y, MAX_ABS_VEL_Y)
        acc_x = normalize(acc_x, MIN_ABS_ACC_X, MAX_ABS_ACC_X)
        acc_y = normalize(acc_y, MIN_ABS_ACC_Y, MAX_ABS_ACC_Y)
        return np.array([x, y, vel_x, vel_y, acc_x, acc_y])


    def generate_data_tensors(self, uuid_to_track):
        self.state_tensors = np.zeros((self.num_frames, 3, 3, self.state_len))
        self.edge_type_tensors = np.zeros((self.num_frames, 3, 3, len(INTERACTION_TYPE_TO_ONEHOT)))
        self.graph_masks = np.zeros((self.num_frames, 3, 3))

        for frame_idx in range(self.num_frames):
            state_tensor_slice = self.state_tensors[frame_idx]
            edge_type_tensor_slice = self.edge_type_tensors[frame_idx]
            graph_masks_slice = self.graph_masks[frame_idx]

            frame = self.frames[frame_idx]
            state = self.get_state(frame_idx) # center, Frenet
            state = self.normalize_state(state)
            state_tensor_slice[1, 1] = state
            # 'n' prefix for 'neighbor'
            neighbors = self.neighbors[:, frame_idx]
            for neighbor_idx, n_uuid in enumerate(neighbors):
                if n_uuid is not None:
                    n_track = uuid_to_track[n_uuid]
                    n_frame_idx = list(n_track.frames).index(frame)
                    tensor_coords = NEIGHBOR_IDX_TO_COORDS[neighbor_idx]
                    # States
                    n_rel_state = n_track.get_relative_state(n_frame_idx, frame_idx, self)
                    n_rel_state = self.normalize_state(n_rel_state)
                    state_tensor_slice[tensor_coords] = n_rel_state
                    # Edge types
                    edge_type = '-'.join([self.v_type, n_track.v_type])
                    edge_type_tensor_slice[tensor_coords] = INTERACTION_TYPE_TO_ONEHOT[edge_type]
                    # Graph masks
                    graph_masks_slice[tensor_coords] = 1


    def generate_temporal_aux_info_tensors(self):
        '''
            Currently not used
        '''
        self.aux_tensors = np.zeros((self.num_frames, 3))
        for frame_idx in range(self.num_frames):
            self.aux_tensors[frame_idx] = np.array([
                self.thws[frame_idx], self.dhws[frame_idx], self.ttcs[frame_idx]])

    def generate_training_data_segments(self):
        if self.num_frames < MIN_NUM_FRAMES:
            return []

        training_data_segments = []
        n_segments = (self.num_frames - TOTAL_SEQ_LEN) // FORWARD_SHIFT_STRIDE + 1

        for i_seg in range(n_segments):
            # Since a lot of tracks end with a 'lane changing' in process,
            # segmenting tracks from thier end should guarantee more info for
            # lane changing. This won't affect lane keeping tracks.
            end = - i_seg * FORWARD_SHIFT_STRIDE - 1
            start = end - TOTAL_SEQ_LEN + 1
            input_pred_split = start + INPUT_SEQ_LEN
            input_seq = self.state_tensors[start : input_pred_split]
            input_masks = self.graph_masks[start : input_pred_split]
            input_edge_types = self.edge_type_tensors[start : input_pred_split]
            if end == -1:
                pred_seq = self.state_tensors[input_pred_split:]
            else:
                pred_seq = self.state_tensors[input_pred_split : input_pred_split + PRED_SEQ_LEN]
            data_point = (input_seq, input_masks, input_edge_types, pred_seq)
            training_data_segments.append(data_point)

        if start != -self.num_frames:
            # One additional data_point starting from 0 to make up for the mis-alignment
            input_seq = self.state_tensors[:INPUT_SEQ_LEN]
            input_masks = self.graph_masks[:INPUT_SEQ_LEN]
            input_edge_types = self.edge_type_tensors[:INPUT_SEQ_LEN]
            pred_seq = self.state_tensors[INPUT_SEQ_LEN : INPUT_SEQ_LEN + PRED_SEQ_LEN]
            data_point = (input_seq, input_masks, input_edge_types, pred_seq)
            training_data_segments.append(data_point)

        return training_data_segments