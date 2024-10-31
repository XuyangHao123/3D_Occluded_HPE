import numpy as np
import torch
import functools
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')


def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def draw_skeleton_2d(data, label=None, color=None, img=None, save=False, gs=None, if_label=False):
    parent = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    limb = []
    if gs:
        ax = plt.subplot(gs)
    else:
        ax = plt.subplot()
    for i in range(len(data)):
        limb.append(np.stack([data[i], data[parent[i]]], axis=1))
    if type(img) == np.ndarray:
        ax.imshow(img)
    for i in range(len(data)):
        if i == len(data)-1 and label and if_label:
            ax.plot(limb[i][0], -limb[i][1], label=label, color=color[i], linewidth=2.0)
        else:
            ax.plot(limb[i][0], -limb[i][1], color=color[i], linewidth=2.0)

    #for j in range(len(data)):
    #    plt.annotate(str(j), (data[j, 0], data[j, 1]), (data[j, 0] + 0.005, data[j, 1] + 0.005), color=color[j])
    for i in range(len(data)):
        ax.scatter(data[i, 0], -data[i, 1], color=color[i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if type(img) != np.ndarray:
        plt.gca().set_aspect('equal')
    '''if not gs:
        plt.show()'''


def draw_skeleton_3d(data, color=None, save=False):
    parent = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    ax = plt.subplot(projection='3d')
    limb = []
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for i in range(len(data)):
        ax.scatter3D(data[i, 0], data[i, 2], data[i, 1], color=color[i])
    for i in range(len(data)):
        limb.append(np.stack([data[i], data[parent[i]]], axis=1))
    for i in range(len(data)):
        ax.plot(limb[i][0], limb[i][2], limb[i][1], color=color[i], linewidth=2.0)
    '''for i in range(len(data)):
        ax.text(data[i][0], data[i][2], -data[i][1], str(i))'''
    plt.gca().set_aspect('equal')
    plt.show()


def perspective_projection(pose_3d):
    p2d = pose_3d[:, :, :2] / pose_3d[:, :, [2]]

    return p2d


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 1024), nn.ReLU(),
                         nn.Linear(1024, dims_out))


def recover(score, m):
    batch_size = score.shape[0]
    g_score = torch.ones(size=(batch_size, 17)).to(score)
    for i in range(batch_size):
        g_score[i, m[i]] = score[i]

    return g_score


def get_min_dist_arg(x):
    # b, 17, 2
    x1 = x[:, None]
    x2 = x[None]
    diff = x1 - x2
    distance = diff.norm(dim=-1).sum(dim=-1)
    min_dist, arg_min_dist = torch.topk(-distance, 2, dim=1)
    min_dist = min_dist[:, -1]
    arg_min_dist = arg_min_dist[:, -1]

    return min_dist, arg_min_dist


def get_influence(distance, w=40):
    influence_score = 2 * torch.sigmoid(distance * w)
    #mean_score = influence_score.mean()
    #std_score = influence_score.std()
    return influence_score


def create_new_pose(pose, net=None, mix_weight=0.9):
    batch_size = pose.shape[0]
    new_pose = pose.clone()
    distance, location = get_min_dist_arg(pose)
    if net:
        input_pose = torch.cat([new_pose, new_pose[location]], dim=1).reshape(batch_size, 34*2)
        input_pose = torch.cat([input_pose, distance[:, None]], dim=-1)
        influence_score1 = net(input_pose)
        influence_score2 = get_influence(distance)
        influence_score = mix_weight*influence_score1 + (1-mix_weight)*influence_score2
    else:
        influence_score = get_influence(distance)

    new_pose *= (1 - influence_score[:, None, None])
    new_pose += pose[location] * influence_score[:, None, None]

    return new_pose
