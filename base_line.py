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
import os
import random
import time
import cv2
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
import tools.fk_helper as fk
import tools.plot_helper as plot
from tools.robot_hands_helper import CommonJoints, JointInfo
from tools.torch_canonical.trans import transform_to_canonical

import sys
sys.path.append("./Frankmocap")
#from Frankmocap import demo
from Frankmocap.demo.demo_handmocap import pic_2_mano
net = torch.load("model/model_100epoch_10082223.pth")
net.cpu().eval()

    # load human hand pose
from datasets.human_hand_dataset import HumanHandDataset
dataset = HumanHandDataset("FreiHAND_pub_v2")

parser = argparse.ArgumentParser()
    

    # type==1 with collision type==0 without collison
parser.add_argument("--epoch", type=int, default=100, help="")
parser.add_argument("--weight",type=float,default=0.1,help="")
parser.add_argument("--learning_rate",type=float,default=0.0001,help="")
parser.add_argument("--batch_size", type=int, default=32, help="")

        
pred_list=pic_2_mano()
single_hand_para=dict()
for hand in pred_list[0][0].keys():
            if pred_list[0][0][hand]==None:
                print(hand +" is None") 
            else:
                single_hand_para=pred_list[0][0][hand]
mano_para=single_hand_para['pred_hand_betas'][0].tolist()+single_hand_para['pred_hand_pose'][0][3:].tolist()

       
        
        # predict robot hand pose
with torch.no_grad():
            robot_joint_position=net(torch.tensor([mano_para]))
            print(robot_joint_position)
            #robot_joint_position = net(vis_roi["mano_input"].view(1, -1))
            chains = fk.get_chains(
                urdf_path="robots/allegro_hand_description/allegro_hand_description_right.urdf", use_gpu=False
            )
            
        # comparing input and prediction in canonical
        
        # set predicted pose in simulation
            physics_client = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
            p.setGravity(0, 0, -9.8)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
            PLANE_ID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
            ROBOT_ID = p.loadURDF(
                os.path.join( "robots", "allegro_hand_description", "allegro_hand_description_right.urdf"),
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
            for index in range(len(movable_joint_ids)):
                p.resetJointState(ROBOT_ID,movable_joint_ids[index],0)
            for index in range(len(movable_joint_ids)):    
                p.resetJointState(ROBOT_ID,movable_joint_ids[index],sim_joint_position[index])
            
            (width, height, rgba, depth, segmentation) = p.getCameraImage(width=640,height=480)    
            img_rgb = rgba[:, :, :3]
            img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            cv2.imwrite('rendered_image.png', img_bgr)  
            