import numpy as np

from datasets.helpers.trajectory_dataset_helper import TrajectoryDataset
from datasets.helpers.fpha_helper import FPHAHelper
import tools.plot_helper as plot


class FPHATrajectoryDataset(TrajectoryDataset):
    def __init__(
            self, data_dir, is_training,  data_aug=False,
            use_norm=False, subjects=None, condition_num=50, future_num=10, frame_step=1, data_max=None, data_min=None
    ):
        super().__init__()
        frame_ids, joints3d = FPHAHelper(data_dir=data_dir, claimed_subjects=subjects).prepare_data()
        joints3d = [FPHATrajectoryDataset.mapping_to_simple(j) for j in joints3d]
        self.past_init(
            frame_ids=frame_ids, joints3d=joints3d, is_training=is_training, data_aug=data_aug, use_norm=use_norm,
            condition_num=condition_num, future_num=future_num, frame_step=frame_step,
            data_max=data_max, data_min=data_min
        )
        self.set_mapping()

    @staticmethod
    def mapping_to_simple(fpha_joints):
        return fpha_joints[..., [
            0,
            1, 6, 7, 8,
            2, 9, 10, 11,
            3, 12, 13, 14,
            4, 15, 16, 17,
            5, 18, 19, 20
        ], :]


if __name__ == "__main__":
    kernel_pcd, kernel_lines = plot.get_plot_graph(
        np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]),
        np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]),
        plot.Color.GREEN
    )
    plot_list = [kernel_pcd, kernel_lines]

    # all together
    subjects = [1, 2, 3, 4, 5, 6]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]]

    joints3d = FPHATrajectoryDataset("..\\Hand_pose_annotation_v1", is_training=True, subjects=subjects, frame_step=2)\
        .joints3d

    apply_plot_list = None
    for i in range(len(joints3d)):
        for t in range(0, joints3d[i].shape[0], 100):
            apply_plot_list = plot_list[:]

            pts = joints3d[i][t, :, :]
            links = plot.Skeleton.HUMAN
            pcd, lines = plot.get_plot_graph(pts, links, colors[0], uniform_color=False)
            apply_plot_list += [pcd, lines]

            # plot
            plot.plot_pts(apply_plot_list)

    # for joint_idx in range(21):
    #     for i in range(len(sequences)):
    #         apply_plot_list = plot_list[:]
    #
    #         formal_pts = formal_joints3d[i][:, joint_idx, :]
    #         formal_links = np.array([[t, t + 1] for t in range(len(formal_pts) - 1)])
    #         formal_pcd, formal_lines = plot.get_plot_graph(formal_pts, formal_links, colors[i], uniform_color=False)
    #         apply_plot_list += [formal_pcd, formal_lines]
    #
    #         normal_pts = normal_joints3d[i][:, joint_idx, :]
    #         normal_links = np.array([[t, t + 1] for t in range(len(normal_pts) - 1)])
    #         normal_pcd, normal_lines = plot.get_plot_graph(normal_pts, normal_links, colors[i], uniform_color=False)
    #         apply_plot_list += [normal_pcd, normal_lines]
    #
    #     # plot
    #     plot.plot_pts(apply_plot_list)
    #     poo = [[ele for ele in dataset.mapping if ele[0] == i] for i in range(len(sequences))]
    print("Done.")
