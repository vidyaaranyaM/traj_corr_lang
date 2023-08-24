import numpy as np


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
    