import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import open3d as o3d
import pybullet as p
import pybullet_data
import torch
from networks.retargeting_mlp import RetargetingMLP
import tools.fk_helper as fk
from losses.energy_loss import anergy_loss_collision_classifier
import tools.plot_helper as plot
from tools.robot_hands_helper import CommonJoints, JointInfo
from tools.torch_canonical.trans import transform_to_canonical
from datasets.human_hand_dataset import HumanHandDataset
# TODO: Too many iterations will freeze the OS
from tools.common import rescale_robot_joint_position
class Angle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.angle = torch.nn.Parameter(torch.zeros(1, 16))  
       

    def forward(self, inputs):
        pass
dataset=HumanHandDataset("FreiHAND_pub_v2")
if torch.cuda.is_available():
        print("using GPU ...")
        device = torch.device("cuda:0")
        chains = fk.get_chains(
            "robots/allegro_hand_description/allegro_hand_description_right.urdf", use_gpu=True, device=device
        )
else:
        print("using CPU ...")
        device = torch.device("cpu")
        chains = fk.get_chains("robots/allegro_hand_description/allegro_hand_description_right.urdf", use_gpu=False)
   
model_classifier=torch.load("model/model_classifier_4n_LeakyRelu0.1_epoch50_datasize10000000_batch256_learningrate0001.pth",map_location=device)

physics_client = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
#p.setGravity(0, 0, -9.8)

PLANE_ID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
ROBOT_ID = p.loadURDF(
                os.path.join( "robots", "allegro_hand_description", "allegro_hand_description_right.urdf"),
                useFixedBase=True,flags=p.URDF_USE_SELF_COLLISION
            )
            # set the center of mass frame (load URDF sets base line frame)
movable_joint_ids = [i for i in range(p.getNumJoints(ROBOT_ID))
                    if i not in CommonJoints.get_joints(is_right=True).TIP]
angles=[]
for i in [301,304,334,353,363]:
    vis_roi = dataset[i]
    human_key_vectors = vis_roi["key_vectors"].to(device=device)            
    angle=Angle()
    optimizer = torch.optim.SGD(angle.parameters(),lr=0.05)
    for i in tqdm.tqdm(range(100)):
        
        energy,loss_collision=anergy_loss_collision_classifier(human_key_vectors=human_key_vectors, robot_joint_angles=angle.angle.to(device), chains=chains,model=model_classifier)

        loss=energy*1000+0.1*loss_collision
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            angle.angle=torch.nn.Parameter(rescale_robot_joint_position(torch.tanh(angle.angle)))


    sim_joint_positions=[]
    for robot_joint_angle in angle.angle:
            joint_position = robot_joint_angle.cpu().detach().view(-1).numpy()
            sim_joint_position = np.zeros_like(joint_position)
            mapping_to_sim = CommonJoints.get_joints(is_right=True).get_mapping_to_sim()
            for i in range(joint_position.shape[0]):
                    sim_joint_position[mapping_to_sim[i]] = joint_position[i]
            sim_joint_positions.append(sim_joint_position)    
    sim_joint_positions=torch.tensor(sim_joint_positions)

    for index in range(len(movable_joint_ids)):
                p.resetJointState(ROBOT_ID,movable_joint_ids[index],0)
            # reset the jointstate
    for index in range(len(movable_joint_ids)):    
                p.resetJointState(ROBOT_ID,movable_joint_ids[index],sim_joint_position[index])
            
    points_pen=p.getClosestPoints(ROBOT_ID,ROBOT_ID,-0.01)
        # points_pen_set.append(points_pen)  
        # get the 3d position of vec
    vec_point=[]
    for point_info in points_pen:
                vec_point.append(point_info[5])
    print(len(vec_point))
    angles.append(angle.angle.detach().numpy())
np.save("result.txt",np.array(angles))