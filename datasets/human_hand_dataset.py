import os

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from tools.commons.io_helper import load_json
from tools.common import get_key_vectors
from tools.plot_helper import plot_joints_to_rgb
from tools.torch_canonical.trans import transform_to_canonical
import torch


class HumanHandDataset(Dataset):
    def __init__(self, dataset_dir, is_training=True):
        super().__init__()

        self.is_training = is_training
        mode = "training" if is_training else "evaluation"

        intrinsics_path = os.path.join(dataset_dir, f"{mode}_K.json")
        scale_path = os.path.join(dataset_dir, f"{mode}_scale.json")

        self.K = torch.tensor(load_json(intrinsics_path))
        self.scale = torch.tensor(load_json(scale_path))

        self.color_path = list()
        for i in range(len(self.scale)):
            title = f"{i:08d}"
            self.color_path.append(os.path.join(dataset_dir, mode, "rgb", f"{title}.jpg"))

        if self.is_training:
            xyz_path = os.path.join(dataset_dir, f"{mode}_xyz.json")
            self.xyz = torch.tensor(load_json(xyz_path))

            self.xyz_from_canonical, self.canonical_from_world = transform_to_canonical(self.xyz, is_human=True)
            self.key_vector = get_key_vectors(self.xyz_from_canonical, is_human=True)

    def __getitem__(self, index):
        color_path = self.color_path[index]
        rgb = plt.imread(color_path)
        if self.is_training:
            roi = {
                "rgb": rgb,  # (224, 224, 3) uint8
                "K": self.K[index],  # (3, 3) float
                "scale": self.scale[index],  # float
                "xyz_input": self.xyz[index].flatten(),  # * (63, ) float
                "xyz": self.xyz[index],  # (21, 3) float
                "key_vectors": self.key_vector[index],  # * (10, 3) float
                "canonical_xyz": self.xyz_from_canonical[index],  # (21, 3) float
                "TCaW": self.canonical_from_world[index],  # (4, 4) float
                # "vert" = self.vert[index],  # point cloud, large data !

            }
        else:
            roi = {
                "rgb": rgb,  # (224, 224, 3) uint8
                "K": self.K[index],  # (3, 3) float
                "scale": self.scale[index],  # float
            }
        return roi

    def __len__(self):
        return len(self.color_path)


if __name__ == "__main__":
    dataset = HumanHandDataset("..//FreiHAND_pub_v2")
    vis_roi = dataset[300]
    plot_joints_to_rgb(rgb=vis_roi["rgb"], intrinsics=vis_roi["K"].numpy(), joints3d=vis_roi["xyz"].numpy())
    joints = vis_roi["xyz"].numpy().copy()
    np.savetxt("../h_joints.txt", joints - joints[0, :])
