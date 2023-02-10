import numpy as np


def cal_inv_mat(mat):
    inv_mat = np.zeros(mat.shape)
    inv_mat[:3, :3], inv_mat[:3, 3] = mat[:3, :3].T, -mat[:3, :3].T @ mat[:3, 3]
    inv_mat[3, 3] = 1
    return inv_mat
