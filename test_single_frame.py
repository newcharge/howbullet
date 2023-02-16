import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pybullet as p
import pybullet_data
import torch
from scipy.spatial.transform import Rotation

import tools.plot_helper as plot
from tools.commons.math_helper import cal_inv_mat
from tools.simulation.robot_hands_helper import CommonJoints, JointInfo
from tools.torch_canonical.trans import transform_to_canonical

# TODO: Too many iterations will freeze the OS
if __name__ == "__main__":
    net = torch.load("model_100epoch_12101231.pth")
    net.cpu().eval()
    net.chains = [c.to(device=torch.device("cpu")) for c in net.chains]

    # load human hand pose
    dataset = torch.load("test_dataset.pth")
    for _ in range(len(dataset)):
        frame_id = random.randint(0, len(dataset))
        vis_roi = dataset[frame_id]
        print(f"choosing frame {frame_id}")
        K = vis_roi["K"].numpy()

        coordinate = vis_roi["xyz"].numpy()
        x2d = K @ coordinate.T
        x2d[:2, :] /= x2d[2, :]
        plt.clf()
        plt.imshow(vis_roi["rgb"])
        plt.plot(x2d[0], x2d[1], ".")
        plt.show()

        # predict robot hand pose
        with torch.no_grad():
            world2canonical = vis_roi["TCaW"]
            robot_joint_position, base2canonical, robot_joints_position_from_canonical = net(
                vis_roi["xyz_input"].view(1, -1)
            )
            canonical_robot_joint_coordinate = robot_joints_position_from_canonical.numpy()[0]
            canonical_human_joint_coordinate = vis_roi["canonical_xyz"].numpy()
            base2world = cal_inv_mat(world2canonical.numpy()) @ base2canonical.numpy()[0]

        # comparing input and prediction in canonical
        human_pcd = plot.get_plot_points(canonical_human_joint_coordinate, plot.Color.BLUE)
        human_lines = plot.get_plot_lines(plot.Skeleton.HUMAN, canonical_human_joint_coordinate, plot.Color.BLUE)
        robot_pcd = plot.get_plot_points(canonical_robot_joint_coordinate, plot.Color.RED)
        robot_lines = plot.get_plot_lines(plot.Skeleton.ROBOT, canonical_robot_joint_coordinate, plot.Color.RED)
        o3d.visualization.draw_geometries([human_pcd, human_lines, robot_pcd, robot_lines], width=1280, height=720)

        # set predicted pose in simulation
        physics_client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setGravity(0, 0, -9.8)

        PLANE_ID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
        ROBOT_ID = p.loadURDF(
            os.path.join("robots", "xarm6_allegro.urdf"),
            useFixedBase=True
        )
        # set the center of mass frame (load URDF sets base line frame)
        p.resetBasePositionAndOrientation(ROBOT_ID, [0, 0, 0.5], p.getQuaternionFromEuler([0, 0, 0]))
        p.resetDebugVisualizerCamera(
            cameraDistance=1,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0.5]
        )

        allegro_joint_ids = [i for i in CommonJoints.get_joints(is_right=True).get_sequence()]
        movable_joint_ids = [i + 1 for i in range(6)] + [
            i for i in allegro_joint_ids
            if i not in CommonJoints.fix_indices(CommonJoints.get_joints(is_right=True).TIP)
        ][1:]
        time_step = 1 / 240
        joint_position = robot_joint_position.detach().view(-1).numpy()

        joint_from_root_position = None
        while True:
            if not p.isConnected():
                break
            try:
                p.stepSimulation()

                target_v = 0
                max_hand_force = 50
                max_arm_force = 500
                target_base_position = base2world[:3, 3].tolist()
                target_base_rotation = Rotation.from_matrix(base2world[:3, :3]).as_quat().tolist()
                target_arm_position = list(p.calculateInverseKinematics(
                    bodyUniqueId=ROBOT_ID,
                    endEffectorLinkIndex=7,
                    targetPosition=target_base_position,
                    targetOrientation=target_base_rotation
                ))[:6]
                p.setJointMotorControlArray(
                    bodyUniqueId=ROBOT_ID,
                    jointIndices=movable_joint_ids,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=target_arm_position + joint_position.tolist(),
                    targetVelocities=[target_v for _ in movable_joint_ids],
                    forces=[max_arm_force for _ in movable_joint_ids[:6]] + [max_hand_force for _ in movable_joint_ids[6:]]
                )
                joint_from_root_position = np.array(list(map(lambda ele: list(ele), [(0, 0, 0)] + [
                    tuple(JointInfo.joint_info(ROBOT_ID, i).get_root_related_position_rotation())[0]
                    for i in CommonJoints.get_joints(is_right=True).get_sequence()[1:]
                ])))

                time.sleep(time_step)
            except p.error as e:
                print(e)
                pass

        # comparing prediction and simulation in canonical
        th_sim_joint_coordinate = torch.tensor(joint_from_root_position).to(dtype=torch.float32, device="cpu")
        th_sim_joint_coordinate, _ = transform_to_canonical(
            th_sim_joint_coordinate.view(1, *th_sim_joint_coordinate.shape), is_human=False
        )
        sim_joint_coordinate = th_sim_joint_coordinate.numpy().reshape(*joint_from_root_position.shape)
        sim_pcd, sim_lines = plot.get_plot_graph(sim_joint_coordinate, plot.Skeleton.ROBOT, plot.Color.GREEN)
        o3d.visualization.draw_geometries([sim_pcd, sim_lines, robot_pcd, robot_lines], width=1280, height=720)

        root2world = base2world @ CommonJoints.T_OR
        x3d = np.zeros((joint_from_root_position.shape[0], 4), dtype=float)
        x3d[:, :3], x3d[:, 3] = joint_from_root_position, 1
        abs_bot_joint_coordinate = (root2world @ x3d.T).T[:, :3]
        abs_bot_pcd, abs_bot_lines = plot.get_plot_graph(
            abs_bot_joint_coordinate, plot.Skeleton.ROBOT, plot.Color.GREEN
        )

        plot.plot_joints_to_rgb(vis_roi["rgb"], vis_roi["K"], abs_bot_joint_coordinate)

        abs_human_joint_coordinate = vis_roi["xyz"].numpy()
        abs_human_pcd, abs_human_lines = plot.get_plot_graph(
            abs_human_joint_coordinate, plot.Skeleton.HUMAN, plot.Color.BLUE
        )

        plot.plot_pts([abs_bot_pcd, abs_bot_lines, abs_human_pcd, abs_human_lines])
