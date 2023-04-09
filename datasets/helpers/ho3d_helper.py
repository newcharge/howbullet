import os
import numpy as np

from tools.commons.io_helper import load_pickle


def _get_suffix(filename):
    return filename.split(".")[-1]


def _transform(joints3d, trans_mat):
    x3d = np.zeros((joints3d.shape[0], 4))
    x3d[:, :3], x3d[:, 3] = joints3d, 1
    trans_joints3d = (x3d @ trans_mat.T)[:, :3]
    return trans_joints3d


class HO3DHelper:
    def __init__(self, data_dir, claimed_sequences=None):
        banned = ["MC1", "ND2", "SiS1", "SM2", "SMu1", "SS1"]
        raw_sequences = os.listdir(data_dir)
        if claimed_sequences is not None:
            allowed_seq = [seq for seq in claimed_sequences if seq in raw_sequences and seq not in banned]
        else:
            allowed_seq = raw_sequences
        self.allowed_seq = allowed_seq
        self.ogl_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]])
        self.data_dir = data_dir  # Usually named as ${any_dirname}/HO3D_v3/train
        self.param_suffix = "pkl"
        self.param_dir_name = "meta"
        self.calibration_dir_name = "calibration"

    def prepare_data(self):
        prepared_joints3d = list()
        all_frame_ids = list()

        for sid in range(len(self.allowed_seq)):
            seq = self.allowed_seq[sid]
            param_dir = os.path.join(self.data_dir, seq, self.param_dir_name)
            trans = self.load_trans(seq)
            filenames = [n for n in os.listdir(param_dir) if _get_suffix(n) == self.param_suffix]
            param_paths = sorted([os.path.join(param_dir, f) for f in filenames], key=HO3DHelper.param2id)
            seq_joints3d = list()
            all_frame_ids.append(list())
            for param_path in param_paths:
                data = load_pickle(param_path)
                joints3d = data["handJoints3D"]
                if joints3d is None:
                    continue
                frame_id = HO3DHelper.param2id(param_path)
                seq_joints3d.append(_transform(joints3d @ self.ogl_mat.T, trans))
                all_frame_ids[sid].append(frame_id)
            seq_joints3d = np.stack(seq_joints3d)
            prepared_joints3d.append(seq_joints3d)

        return all_frame_ids, prepared_joints3d

    def load_trans(self, sequence_name):
        calibrate_root = os.path.join(self.data_dir, "..", self.calibration_dir_name)
        assert sequence_name[:-1] in os.listdir(calibrate_root), \
            "Sequence name should be composed of ExperimentID+CameraOrderID(1 digit), such as ABF10, BB10, etc."
        calibrate_dir = os.path.join(calibrate_root, sequence_name[:-1], "calibration")
        cams_order = np.loadtxt(os.path.join(calibrate_dir, 'cam_orders.txt')).astype('uint8').tolist()
        idx = cams_order.index(int(sequence_name[-1]))
        trans = np.loadtxt(os.path.join(calibrate_dir, f"trans_{idx}.txt"))
        return trans

    # @staticmethod
    # def recenter(joints):
    #     assert len(joints.shape) == 3 and joints.shape[1] == 21 and joints.shape[2] == 3  # joints.shape == [N, 21, 3]
    #     joints_mean = np.mean(np.mean(joints, axis=0), axis=0)
    #     return joints - joints_mean

    @staticmethod
    def param2id(filepath):
        return int(filepath.split(os.path.sep)[-1].split(".")[0])
