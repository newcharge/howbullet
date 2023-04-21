import numpy as np
from scipy.spatial.transform import Rotation


def cal_inv_mat(mat):
    assert mat.shape[0] == 4 and mat.shape[1] == 4, "The input matrix should be 4x4 format !"
    inv_mat = np.zeros(mat.shape)
    inv_mat[:3, :3], inv_mat[:3, 3] = mat[:3, :3].T, -mat[:3, :3].T @ mat[:3, 3]
    inv_mat[3, 3] = 1
    return inv_mat


def transform(mat, points):
    assert points.shape[1] == 3, "The points should be Nx3 format !"
    assert mat.shape[0] == 4 and mat.shape[1] == 4, "The input matrix should be 4x4 format !"
    x3d = np.ones((points.shape[0], 4))
    x3d[:, :3] = points
    trans_x3d = x3d @ mat.T
    return trans_x3d[:, :3]


def transform_from_position_rotation(position, rotation):
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = Rotation.from_quat(rotation).as_matrix()
    trans_mat[:3, 3] = position
    return trans_mat
