import torch

from tools.simulation.robot_hands_helper import CommonJoints


def get_key_point_indices(is_human=True):
    if is_human:
        # 0 is root, 4, 8, 12, 16, 20 are finger tips
        return [0, 4, 8, 12, 16]
    else:
        # 0 is root, 5, 10, 15, 20 are finger tips
        return [0, 5, 10, 15, 20]


def get_key_vectors(joints, is_human):
    assert len(joints.shape) == 3, "joints' shape should be N x 21 x 3"
    key_points = joints[:, get_key_point_indices(is_human), :]
    key_vectors = list()
    for i in range(key_points.shape[1]):
        for j in range(i + 1, key_points.shape[1]):
            key_vectors.append(key_points[:, j, :] - key_points[:, i, :])
    key_vectors = torch.stack(key_vectors, dim=1)
    return key_vectors


def rescale_robot_joint_position(joints):
    assert len(joints.shape) == 2, "joints' shape should be N x 16"
    joint_infos = [ele for ele in CommonJoints.get_joints(is_right=True).LINK_INFOS if not ele["name"].endswith("tip")]
    rescaled_joints = list()
    for i in range(joints.shape[1]):
        info = joint_infos[i]
        lower_bound, upper_bound = info["lower_bound"], info["upper_bound"]
        rescaled_joints.append((joints[:, i] + 1) / 2 * (upper_bound - lower_bound) + lower_bound)
    rescaled_joints = torch.stack(rescaled_joints, dim=1)
    return rescaled_joints
