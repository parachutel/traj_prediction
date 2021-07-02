import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_anneal(steps, start, finish, center_step, steps_lo_to_hi):
    return start + (finish - start) * sigmoid((steps - center_step) * (1. / steps_lo_to_hi))

n_steps = 20000
steps = np.array(range(n_steps))

# kl
kl_start = 0.0001
kl_finish = 1.0
kl_center_step = 8000
kl_sigmoid_divisor = 6
kl_crossover = kl_center_step
kl_steps_lo_to_hi = kl_crossover / kl_sigmoid_divisor

kl_anneal = sigmoid_anneal(steps, kl_start, kl_finish, kl_center_step, kl_steps_lo_to_hi)

plt.plot(steps, kl_anneal, label='kl_weight')
plt.show()