import numpy as np
import random


def convert_deltas(delta):
    if (delta == np.array([0, 0])).all():
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif (delta == np.array([-0.01, 0])).all():
        return np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif (delta == np.array([-0.01, 0.01])).all():
        return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif (delta == np.array([0, 0.01])).all():
        return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif (delta == np.array([0.01, 0.01])).all():
        return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif (delta == np.array([0.01, 0])).all():
        return np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif (delta == np.array([0.01, -0.01])).all():
        return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif (delta == np.array([0, -0.01])).all():
        return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif (delta == np.array([-0.01, -0.01])).all():
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

def convert_actions(action):
    if (action == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])).all():
        return np.array([0, 0])
    elif (action == np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])).all():
        return np.array([-0.01, 0])
    elif (action == np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])).all():
        return np.array([-0.01, 0.01])
    elif (action == np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])).all():
        return np.array([0, 0.01])
    elif (action == np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])).all():
        return np.array([0.01, 0.01])
    elif (action == np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])).all():
        return np.array([0.01, 0])
    elif (action == np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])).all():
        return np.array([0.01, -0.01])
    elif (action == np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])).all():
        return np.array([0, -0.01])
    elif (action == np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])).all():
        return np.array([-0.01, -0.01])

def sample_delta(idx):
    if idx == 1:
        new_idx = random.choice([8, 2])
    elif idx == 2:
        new_idx = random.choice([1, 3])
    elif idx == 3:
        new_idx = random.choice([2, 4])
    elif idx == 4:
        new_idx = random.choice([3, 5])
    elif idx == 5:
        new_idx = random.choice([4, 6])
    elif idx == 6:
        new_idx = random.choice([5, 7])
    elif idx == 7:
        new_idx = random.choice([6, 8])
    elif idx == 8:
        new_idx = random.choice([7, 1])
    action = np.zeros(9,)
    action[new_idx] = 1
    delta = convert_actions(action)
    return delta, action
