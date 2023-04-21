import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data_max, self.data_min = None, None
        self.mapping = None
        self.frame_ids = None
        self.joints3d = None
        self.norm_joints3d = None
        self.is_training = True
        self.data_aug = False
        self.use_norm = False
        self.condition_num = 0
        self.future_num = 0
        self.frame_step = 0

    def past_init(
            self, frame_ids, joints3d, is_training, data_aug, condition_num, future_num, frame_step, data_max, data_min, use_norm
    ):
        self.mapping = list()
        self.frame_ids = frame_ids
        self.joints3d = joints3d
        self.condition_num = condition_num
        self.future_num = future_num
        self.frame_step = frame_step
        self.is_training = is_training
        self.data_aug = data_aug
        self.use_norm = use_norm
        if is_training:
            self.data_max = max([np.max(s) for s in self.joints3d])
            self.data_min = min([np.min(s) for s in self.joints3d])
            if use_norm:
                self.normalize()
        else:
            self.data_max, self.data_min = data_max, data_min

    def __getitem__(self, index):
        sid, end_idx = self.mapping[index]
        ids = [end_idx - self.frame_step * n for n in range(self.condition_num + self.future_num)][::-1]
        condition_ids, future_ids = ids[:self.condition_num], ids[self.condition_num:]
        condition_frame = torch.from_numpy(
            self.joints[sid][condition_ids].reshape(len(condition_ids), -1).astype(np.float32)
        )
        future_frame = torch.from_numpy(
            self.joints[sid][future_ids].reshape(len(future_ids), -1).astype(np.float32)
        )
        if self.data_aug and torch.rand(1)[0] < .5:
                flipped_frame = torch.cat([condition_frame, future_frame], dim=0)[::-1, ...]
                condition_frame = flipped_frame[:condition_frame.shape[0], ...]
                future_frame = flipped_frame[condition_frame.shape[0]:, ...]
        roi = {
            "condition_frame": condition_frame,
            "future_frame": future_frame
        }
        return roi

    def __len__(self):
        return len(self.mapping)

    def set_mapping(self):
        for sid in range(len(self.frame_ids)):
            start_idx = -1
            for idx in range(len(self.frame_ids[sid])):
                if idx == 0:
                    start_idx = 0
                if 1 <= idx and self.frame_ids[sid][-2] != self.frame_ids[sid][-1] - 1:
                    start_idx = idx
                if self.condition_num + self.future_num <= (idx - start_idx) // self.frame_step + 1:
                    self.mapping.append((sid, idx))

    def get_max_min(self):
        return self.data_max, self.data_min

    def normalize(self):
        self.norm_joints3d = list()
        # recenter
        for idx in range(len(self.joints3d)):
            joints = self.joints3d[idx]
            assert len(joints.shape) == 3 and joints.shape[1] == 21 and joints.shape[2] == 3  # joints.shape: [N, 21, 3]
            joints_mean = np.mean(np.mean(joints, axis=0), axis=0)
            self.norm_joints3d.append(joints - joints_mean)
        # normalize
        self.norm_joints3d = [2 * (s - self.data_min) / (self.data_max - self.data_min) - 1 for s in self.joints3d]

    @property
    def joints(self):
        return self.norm_joints3d if self.is_training and self.use_norm else self.joints3d

    @staticmethod
    def mapping_to_simple(mano_joints):
        return mano_joints[..., range(len(mano_joints.shape[-2])), :]
