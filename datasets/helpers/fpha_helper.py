import os

import numpy as np


class FPHAHelper:
    def __init__(self, data_dir, claimed_subjects=None):
        banned = list()
        raw_sequences = os.listdir(data_dir)
        if claimed_subjects is not None:
            allowed_sub = [
                sub for sub in claimed_subjects if FPHAHelper.sub_dir_name(sub) in raw_sequences and sub not in banned
            ]
        else:
            allowed_sub = raw_sequences
        self.allowed_sub = allowed_sub
        self.data_dir = data_dir  # Usually named as ${any_dirname}/Hand_pose_annotation_v1
        self.param_name = "skeleton.txt"

    def prepare_data(self):
        all_frame_ids = list()
        joints3d = list()

        for sub_id in sorted([int(i) for i in self.allowed_sub]):
            sub_dir = os.path.join(self.data_dir, FPHAHelper.sub_dir_name(sub_id))
            for act in sorted(os.listdir(sub_dir)):
                act_dir = os.path.join(sub_dir, act)
                for seq in sorted(os.listdir(act_dir)):
                    seq_dir = os.path.join(act_dir, seq)
                    param_path = os.path.join(seq_dir, self.param_name)
                    data = np.loadtxt(param_path)
                    if data.shape[0] == 0:
                        continue
                    seq_joints3d = data[:, 1:] * 1e-3  # mm -> m
                    seq_joints3d = seq_joints3d.reshape(seq_joints3d.shape[0], -1, 3)
                    all_frame_ids.append([i for i in range(seq_joints3d.shape[0])])
                    joints3d.append(seq_joints3d)
        return all_frame_ids, joints3d

    @staticmethod
    def sub_dir_name(subject_id):
        return f"Subject_{subject_id}"
