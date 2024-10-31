import numpy as np
import torch
from utils import draw_skeleton_2d


def normalize_head(poses_2d, root_joint=0):
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [root_joint]]

    scale = np.linalg.norm(p2d[:, :, [0]] - p2d[:, :, [10]], axis=1, keepdims=True).mean()
    '''scale_mean = scale.mean()
    scale_std = scale.std()
    idx = np.argmax(scale[:, 0])
    pose = p2d[idx].T
    draw_skeleton_2d(p2d[idx].T)'''
    p2ds = poses_2d / scale.mean()

    p2ds = p2ds * (1 / 10)

    return p2ds

# 145.5329587164913 human3.6
# 0.21418024948394107 mpii projection
# 359.00276381816576 mpii scaled
# 362.7647966761919 occ


def normalize_head_test(poses_2d, scale=145.5329587164913):  # ground truth
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [0]]

    p2ds = poses_2d / scale
    p2ds = p2ds * (1 / 10)

    return p2ds
