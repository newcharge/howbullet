import numpy as np

from datasets.helpers.ho3d_helper import HO3DHelper
from datasets.helpers.trajectory_dataset_helper import TrajectoryDataset
import tools.plot_helper as plot


class HO3DTrajectoryDataset(TrajectoryDataset):
    def __init__(
            self, data_dir, is_training, data_aug=False,
            use_norm=False, sequences=None, condition_num=50, future_num=10, frame_step=1, data_max=None, data_min=None
    ):
        super().__init__()
        frame_ids, joints3d = HO3DHelper(data_dir=data_dir, claimed_sequences=sequences)\
            .prepare_data()
        joints3d = [HO3DTrajectoryDataset.mapping_to_simple(j) for j in joints3d]
        self.past_init(
            frame_ids=frame_ids, joints3d=joints3d, is_training=is_training, data_aug=data_aug, use_norm=use_norm,
            condition_num=condition_num, future_num=future_num, frame_step=frame_step,
            data_max=data_max, data_min=data_min
        )
        self.set_mapping()

    @staticmethod
    def mapping_to_simple(mano_joints):
        return mano_joints[..., [
            0,
            13, 14, 15, 16,
            1, 2, 3, 17,
            4, 5, 6, 18,
            10, 11, 12, 19,
            7, 8, 9, 20
        ], :]


if __name__ == "__main__":
    kernel_pcd, kernel_lines = plot.get_plot_graph(
        np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]),
        np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]),
        plot.Color.GREEN
    )
    plot_list = [kernel_pcd, kernel_lines]

    # all together
    sequences = [
        "ABF10", "BB10", "GPMF10", "GSF10", "MDF10", "SB10", "ShSu10",
        "SiBF10", "SMu40", "MC1", "ND2", "SiS1", "SM2", "SMu1", "SS1"
    ]
    colors = [
        [1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1],
        [.5, .5, .5], [1, 0, .5], [1, .5, 0], [0, 1, .5], [1, 0, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [.5, .5, .5]
    ]
    # sequences = ["SiBF10", "SiBF11", "SiBF12", "SiBF13", "SiBF14"]
    # colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]]
    # sequences = ["ABF10"]
    # colors = [[1, 0, 0]]

    dataset = HO3DTrajectoryDataset("..\\HO3D_v3\\train", is_training=True, sequences=sequences, frame_step=2)
    data_max, data_min = dataset.get_max_min()
    normal_joints3d = dataset.norm_joints3d
    # normal_joints3d = dataset.recenter_joints3d  # discard normalization
    formal_joints3d = HO3DTrajectoryDataset(
        "..\\HO3D_v3\\train", is_training=False, sequences=sequences, frame_step=2
    ).joints3d

    apply_plot_list = None
    # for i in range(len(sequences)):
    #     for t in range(0, formal_joints3d[i].shape[0], 100):
    #         apply_plot_list = plot_list[:]
    #
    #         formal_pts = HO3DTrajectoryDataset.mapping_to_simple(formal_joints3d[i][t, :, :])
    #         formal_links = plot.Skeleton.HUMAN
    #         formal_pcd, formal_lines = plot.get_plot_graph(formal_pts, formal_links, colors[i], uniform_color=False)
    #         apply_plot_list += [formal_pcd, formal_lines]
    #
    #         normal_pts = HO3DTrajectoryDataset.mapping_to_simple(normal_joints3d[i][t, :, :])
    #         normal_links = plot.Skeleton.HUMAN
    #         normal_pcd, normal_lines = plot.get_plot_graph(normal_pts, normal_links, colors[i], uniform_color=False)
    #         apply_plot_list += [normal_pcd, normal_lines]
    #
    #         # plot
    #         plot.plot_pts(apply_plot_list)
    #         poo = [[ele for ele in dataset.mapping if ele[0] == i] for i in range(len(sequences))]

    joint_idx = 0
    for i in range(len(sequences)):
        apply_plot_list = plot_list[:]

        formal_pts = formal_joints3d[i][:, joint_idx, :]
        formal_links = np.array([[t, t + 1] for t in range(len(formal_pts) - 1)])
        formal_pcd, formal_lines = plot.get_plot_graph(formal_pts, formal_links, colors[i], uniform_color=False)
        apply_plot_list += [formal_pcd, formal_lines]

        normal_pts = normal_joints3d[i][:, joint_idx, :]
        normal_links = np.array([[t, t + 1] for t in range(len(normal_pts) - 1)])
        normal_pcd, normal_lines = plot.get_plot_graph(normal_pts, normal_links, colors[i], uniform_color=False)
        apply_plot_list += [normal_pcd, normal_lines]

        # plot
        plot.plot_pts(apply_plot_list)
        poo = [[ele for ele in dataset.mapping if ele[0] == i] for i in range(len(sequences))]
    print("Done.")
