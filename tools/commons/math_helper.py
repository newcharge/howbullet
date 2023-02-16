import numpy as np


def cal_inv_mat(mat):
    assert mat.shape[0] == 4 and mat.shape[1] == 4, "The input matrix should be 4x4 format !"
    inv_mat = np.zeros(mat.shape)
    inv_mat[:3, :3], inv_mat[:3, 3] = mat[:3, :3].T, -mat[:3, :3].T @ mat[:3, 3]
    inv_mat[3, 3] = 1
    return inv_mat
