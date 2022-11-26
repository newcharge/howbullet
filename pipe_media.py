import cv2
import mediapipe as mp
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
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--input_path", type=str, default="test_for_hand.mp4", help="")
parser.add_argument("--output_path", type=str, default="output_for_hand.mp4", help="")
parser.add_argument("--model_path", type=str, default=None, help="")
# type==0 with collision  type==1 without collision
args = parser.parse_args() 
if args.model_path==None:
    print("model path doesn't exist")
    assert(0)
cap = cv2.VideoCapture(args.input_path)
frames_num = cap.get(7)
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
handConsStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)
pTime = 0
cTime = 0
Results=[]
f=0
while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        
        if result.multi_hand_landmarks:
           
            for handLms in result.multi_hand_landmarks:
                res=[]
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConsStyle)
                for i, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    zPos = int(lm.z )
                    
                    res.append((lm.x, lm.y,lm.z))
                    cv2.putText(img, str(i), (xPos - 25, yPos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    
            if f%1==0:       
                Results.append(res)                            
            f+=1           
    else:
        break   
model_path=args.model_path
#path="epoch_100_e_"+args.e+"_coodi_model.pth"
#path="epoch_100_e_"+args.e+"_21coodi_model.pth"
#path="epoch_100_e__21coodi_Nocollision_model.pth"
#net = torch.load("model\\100coodi_new_model.pth",map_location=torch.device('cpu'))
net = torch.load(model_path,map_location=torch.device('cpu'))
Sim_Joint_positions=[]
    # load human hand pose
for i in Results:
    Coordin=torch.tensor(i)
    #use 21 cood
    robot_joint_position = net(Coordin.view(1,-1))
    chains = fk.get_chains(
                urdf_path="robots/allegro_hand_description/allegro_hand_description_right.urdf", use_gpu=False
            )
    joint_position = robot_joint_position.detach().view(-1).numpy()
    sim_joint_position = np.zeros_like(joint_position)
    mapping_to_sim = CommonJoints.get_joints(is_right=True).get_mapping_to_sim()
    for i in range(joint_position.shape[0]):
            sim_joint_position[mapping_to_sim[i]] = joint_position[i]
    Sim_Joint_positions.append(sim_joint_position)
    joint_from_root_position = None
physics_client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setGravity(0, 0, -9.8)
log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, args.output_path)

PLANE_ID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
ROBOT_ID = p.loadURDF(os.path.join("..", "robots", "allegro_hand_description", "allegro_hand_description_right.urdf"),
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
frames_num = cap.get(7)
rate=cap.get(5)
time_step = 0.00003
number=0
count=0
num_move=250
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
                    targetPositions=Sim_Joint_positions[int(count/num_move)].tolist(),
                    targetVelocities=[target_v for _ in movable_joint_ids],
                    forces=[max_force for _ in movable_joint_ids]
                )
                
                if int(count/num_move)>=len(Sim_Joint_positions)-1:
                    print("finish!")
                    break
                else:
                    count+=1
            except p.error as e:
                print(e)
                pass
p.stopStateLogging(log_id)