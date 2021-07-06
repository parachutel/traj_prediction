import torch
import matplotlib.pyplot as plt
import os
file_path = os.path.dirname(os.path.abspath(__file__))

TORCH_TENSOR_PATH = file_path + '/../../data/processed_data/highd/{}_{}.pt'

from track import *

data_id = 3
traj_id = 2342


data_str = '{:02d}'.format(data_id)
input_seq = torch.load(TORCH_TENSOR_PATH.format(data_str, 'input_seq'))
# input_masks = torch.load(TORCH_TENSOR_PATH.format(data_str, 'input_masks'))
# input_edge_types = torch.load(TORCH_TENSOR_PATH.format(data_str, 'input_edge_types'))
pred_seq = torch.load(TORCH_TENSOR_PATH.format(data_str, 'pred_seq'))

input_seq = input_seq.numpy()
pred_seq = pred_seq.numpy()



input_traj = input_seq[traj_id, :, 1, 1, :2]
pred_traj = pred_seq[traj_id, :, 1, 1, :2]

plt.plot(denormalize(input_traj[:, 0], MIN_X, MAX_X), input_traj[:, 1] * 4, marker='o')
plt.plot(denormalize(pred_traj[:, 0], MIN_X, MAX_X), pred_traj[:, 1] * 4, marker='o')

ground_truth_vels = pred_seq[traj_id, :, 1, 1, 2:4]
dt = 1/25

x_t, y_t = denormalize(input_traj[-1, 0], MIN_X, MAX_X), input_traj[-1, 1] * 4
traj_x_t, traj_y_t = [], []
for t in range(50):
    x_dot_t, y_dot_t = ground_truth_vels[t]
    x_dot_t = denormalize(x_dot_t, MIN_ABS_VEL_X, MAX_ABS_VEL_X)
    y_dot_t = denormalize(y_dot_t, MIN_ABS_VEL_Y, MAX_ABS_VEL_Y)
    x_t = x_t + dt * x_dot_t
    y_t = y_t + dt * y_dot_t
    traj_x_t.append(x_t)
    traj_y_t.append(y_t)
plt.plot(traj_x_t, traj_y_t, color='green', marker='o')

plt.show()