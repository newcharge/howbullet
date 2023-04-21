import os

import numpy as np


class ARCTICHelper:
    def __init__(self, data_dir, split):  # sequences are named such as s%d/%s
        self.data_dir = data_dir  # Usually named as ${any_dirname}/splits
        self.split = split
        self.file_path = os.path.join(data_dir, f"{split}.npy")

    def prepare_data(self):
        all_frame_ids = list()
        joints3d = list()

        data = np.load(self.file_path, allow_pickle=True).item()
        for seq in data["data_dict"]:
            joints_data = data["data_dict"][seq]["cam_coord"]["joints.right"]
            for i in range(joints_data.shape[1]):
                joints3d.append(joints_data[:, i, ...])
                all_frame_ids.append(range(joints_data.shape[0]))
        return all_frame_ids, joints3d
