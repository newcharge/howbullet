import os

import numpy as np
import torch
from torch.utils.data import Dataset

from tools.commons.io_helper import load_pickle
import tools.plot_helper as plot


def _get_suffix(filename):
    return filename.split(".")[-1]


def _recenter(joints):
    assert len(joints.shape) == 3 and joints.shape[1] == 21 and joints.shape[2] == 3  # joints.shape == [N, 21, 3]
    joints_mean = np.mean(np.mean(joints, axis=0), axis=0)
    return joints - joints_mean


def _load_trans(dataset_dir, sequence_name):
    calibrate_root = os.path.join(dataset_dir, "..", "calibration")
    assert sequence_name[:-1] in os.listdir(calibrate_root), \
        "Sequence name should be composed of ExperimentID+CameraOrderID(1 digit), such as ABF10, BB10, etc."
    calibrate_dir = os.path.join(calibrate_root, sequence_name[:-1], "calibration")
    cams_order = np.loadtxt(os.path.join(calibrate_dir, 'cam_orders.txt')).astype('uint8').tolist()
    idx = cams_order.index(int(sequence_name[-1]))
    trans = np.loadtxt(os.path.join(calibrate_dir, f"trans_{idx}.txt"))
    return trans


def _transform(joints3d, trans_mat):
    x3d = np.zeros((joints3d.shape[0], 4))
    x3d[:, :3], x3d[:, 3] = joints3d, 1
    trans_joints3d = (x3d @ trans_mat.T)[:, :3]
    return trans_joints3d


def mapping_to_simple(mano_joints):
    return mano_joints[[
        0,
        13, 14, 15, 16,
        1, 2, 3, 17,
        4, 5, 6, 18,
        10, 11, 12, 19,
        7, 8, 9, 20
    ]]


class TrajectoryDataset(Dataset):
    def __init__(self, dataset_dir, is_training=True, claimed_sequences=None, data_max=None, data_min=None, step=1):
        super().__init__()
        self.formal_joints3d = list()
        self.recenter_joints3d = list()
        self.is_training = is_training
        self.mapping = list()
        ogl_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]])

        banned = list()
        raw_sequences = os.listdir(dataset_dir)
        if claimed_sequences is not None:
            allowed_seq = [seq for seq in claimed_sequences if seq in raw_sequences and seq not in banned]
        else:
            allowed_seq = raw_sequences

        for sid in range(len(allowed_seq)):
            seq = allowed_seq[sid]
            param_dir = os.path.join(dataset_dir, seq, "meta")
            suffix = "pkl"
            trans = _load_trans(dataset_dir, seq)
            filenames = [n for n in os.listdir(param_dir) if _get_suffix(n) == suffix]
            param_paths = sorted([os.path.join(param_dir, f) for f in filenames])
            seq_joints3d = list()
            visited = list()
            for idx, param_path in enumerate(param_paths):
                data = load_pickle(param_path)
                joints3d = data["handJoints3D"]
                if joints3d is None:
                    continue
                frame_id = int(param_path.split(os.path.sep)[-1].split(".")[0])
                visited.append(frame_id)
                seq_joints3d.append(_transform(joints3d @ ogl_mat.T, trans))
                if frame_id - step in visited:
                    self.mapping.append((sid, visited.index(frame_id - step), len(visited) - 1))
            seq_joints3d = np.stack(seq_joints3d)
            self.formal_joints3d.append(seq_joints3d)
            self.recenter_joints3d.append(_recenter(seq_joints3d))
        self.data_max, self.data_min = data_max, data_min  # max-min
        if is_training:
            self.data_max = max([np.max(s) for s in self.recenter_joints3d])
            self.data_min = min([np.min(s) for s in self.recenter_joints3d])

    def __getitem__(self, index):
        joints3d = self.recenter_joints3d
        sid, condition, future = self.mapping[index]
        condition_frame = torch.from_numpy(joints3d[sid][condition].astype(np.float32))
        future_frame = torch.from_numpy(joints3d[sid][future].astype(np.float32))
        roi = {
            "condition_frame": condition_frame,
            "future_frame": future_frame
        }
        return roi

    def __len__(self):
        return len(self.mapping)

    def get_max_min(self):
        return self.data_max, self.data_min


if __name__ == "__main__":
    kernel_pcd, kernel_lines = plot.get_plot_graph(
        np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]),
        np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]),
        plot.Color.GREEN
    )
    plot_list = [kernel_pcd, kernel_lines]

    # all together
    # sequences = [
    #     "ABF10", "BB10", "GPMF10", "GSF10", "MDF10", "SB10", "ShSu10",
    #     "SiBF10", "SMu40", "MC1", "ND2", "SiS1", "SM2", "SMu1", "SS1"
    # ]
    # colors = [
    #     [1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1],
    #     [.5, .5, .5], [1, 0, .5], [1, .5, 0], [0, 1, .5], [1, 0, 0],
    #     [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [.5, .5, .5]
    # ]
    sequences = ["SiBF10", "SiBF11", "SiBF12", "SiBF13", "SiBF14"]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]]
    # sequences = ["ABF10"]
    # colors = [[1, 0, 0]]

    dataset = TrajectoryDataset("..\\HO3D_v3\\train", is_training=True, claimed_sequences=sequences, step=2)
    data_max, data_min = dataset.get_max_min()
    normal_joints3d = [2 * (s - data_min) / (data_max - data_min) - 1 for s in dataset.recenter_joints3d]
    formal_joints3d = dataset.formal_joints3d

    for i in range(len(sequences)):
        # formal_pts = formal_joints3d[i][:, 0, :]
        # formal_links = np.array([[t, t + 1] for t in range(len(formal_pts) - 1)])
        # formal_pcd, formal_lines = plot.get_plot_graph(formal_pts, formal_links, colors[i], uniform_color=False)
        # plot_list += [formal_pcd, formal_lines]

        normal_pts = normal_joints3d[i][:, 0, :]
        normal_links = np.array([[t, t + 1] for t in range(len(normal_pts) - 1)])
        normal_pcd, normal_lines = plot.get_plot_graph(normal_pts, normal_links, colors[i], uniform_color=False)
        plot_list += [normal_pcd, normal_lines]

    # plot
    plot.plot_pts(plot_list)
    poo = [[ele for ele in dataset.mapping if ele[0] == i] for i in range(len(sequences))]
    print("Done.")
