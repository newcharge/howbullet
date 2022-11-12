import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pybullet as p
import pybullet_data
import torch

import tools.fk_helper as fk
import tools.plot_helper as plot
from tools.robot_hands_helper import CommonJoints, JointInfo
from tools.torch_canonical.trans import transform_to_canonical

# TODO: Too many iterations will freeze the OS
if __name__ == "__main__":
    net = torch.load("epoch_100_e_0.003_coodi_model.pth",map_location=torch.device('cpu'))
    net.cpu().eval()

    # load human hand pose
    dataset = torch.load("test_dataset.pth")
    compare_num=0
    for _ in range(len(dataset)):
        frame_id = random.randint(0, len(dataset))
        #vis_roi = dataset[frame_id]
        #print(f"choosing frame {frame_id}")
        vis_roi = dataset[compare_num]
        compare_num+=1
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

            robot_joint_position = net(vis_roi["xyz"][[0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]].view(1,-1))
            #robot_joint_position = net(vis_roi["mano_input"][0:45].view(1, -1))
            chains = fk.get_chains(
                urdf_path="robots/allegro_hand_description/allegro_hand_description_right.urdf", use_gpu=False
            )
            th_joint_coordinates = fk.robot_joint_angle_to_coordinate(chains, robot_joint_position)
            robot_joints_position_from_canonical, _ = transform_to_canonical(th_joint_coordinates, is_human=False)
            robot_joint_coordinate = robot_joints_position_from_canonical.numpy()[0]
            human_joint_coordinate = vis_roi["canonical_xyz"].numpy()

        # comparing input and prediction in canonical
        human_pcd = plot.get_plot_points(human_joint_coordinate, plot.Color.BLUE)
        human_lines = plot.get_plot_lines(plot.Skeleton.HUMAN, human_joint_coordinate, plot.Color.BLUE)
        robot_pcd = plot.get_plot_points(robot_joint_coordinate, plot.Color.RED)
        robot_lines = plot.get_plot_lines(plot.Skeleton.ROBOT, robot_joint_coordinate, plot.Color.RED)
        o3d.visualization.draw_geometries([human_pcd, human_lines, robot_pcd, robot_lines], width=1280, height=720)

        # set predicted pose in simulation
        physics_client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setGravity(0, 0, -9.8)

        PLANE_ID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
        ROBOT_ID = p.loadURDF(
            os.path.join("..", "robots", "allegro_hand_description", "allegro_hand_description_right.urdf"),
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

        movable_joint_ids = [i for i in range(p.getNumJoints(ROBOT_ID))
                             if i not in CommonJoints.get_joints(is_right=True).TIP]

        time_step = 1 / 240
        joint_position = robot_joint_position.detach().view(-1).numpy()
        sim_joint_position = np.zeros_like(joint_position)
        mapping_to_sim = CommonJoints.get_joints(is_right=True).get_mapping_to_sim()
        for i in range(joint_position.shape[0]):
            sim_joint_position[mapping_to_sim[i]] = joint_position[i]

        joint_from_root_position = None
        #p.setRealTimeSimulation(0)
        while True:
            if not p.isConnected():
                break
            try:
                p.stepSimulation()
                target_v = 0
                max_force = 5
                p.setJointMotorControlArray(
                    bodyUniqueId=ROBOT_ID,
                    jointIndices=movable_joint_ids,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=sim_joint_position.tolist(),
                    targetVelocities=[target_v for _ in movable_joint_ids],
                    forces=[max_force for _ in movable_joint_ids]
                )
                
                infos=p.getJointStates(bodyUniqueId=ROBOT_ID,jointIndices=movable_joint_ids)
                Joint_Pos_infos=[]
                for info in infos:
                    Joint_Pos_infos.append(info[0])
                x=np.array(Joint_Pos_infos)
                y=np.array(sim_joint_position)
                '''
                if (abs((x-y)/y)).max()<0.01:
                    print("###################reach!!!###################")
                    print("targetPositions:")
                    print(sim_joint_position)
                    print("getJointStates:")
                    print(Joint_Pos_infos)
                ''' 
                joint_from_root_position = np.array(list(map(lambda ele: list(ele), [(0, 0, 0)] + [
                    tuple(JointInfo.joint_info(ROBOT_ID, i - 1).get_root_related_position_rotation())[0]
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
