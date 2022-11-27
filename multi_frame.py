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
parser.add_argument("--e", type=str, default=None, help="")
args = parser.parse_args()   
cap = cv2.VideoCapture(r"test_for_hand.mp4")# mp4 path
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
                    #if i == 4:
                        #cv2.circle(img, (xPos, yPos), 20, (0, 0, 255), cv2.FILLED)
            if f%1==0:       
                Results.append(res)
                #print(res)

            cTime = time.time()
            fps = 1/(cTime - pTime)
        
            pTime = cTime
                              # 图像，      文字内容，      坐标(右上角坐标)，字体， 大小，  颜色，    字体厚度
            #cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            f+=1
        #cv2.imshow('img', img)
   
    else:
        break   
    #if cv2.waitKey(1) == ord('q'):
        #break
path="epoch_100_e_"+args.e+"_coodi_model.pth"
#net = torch.load("model\\100coodi_new_model.pth",map_location=torch.device('cpu'))
net = torch.load(path,map_location=torch.device('cpu'))
Sim_Joint_positions=[]
    # load human hand pose
for i in Results:
    Coordin=torch.tensor(i)
    robot_joint_position = net(Coordin[[0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]].view(1,-1))
    chains = fk.get_chains(
                urdf_path="robots/allegro_hand_description/allegro_hand_description_right.urdf", use_gpu=False
            )
    #th_joint_coordinates = fk.robot_joint_angle_to_coordinate(chains, robot_joint_position)
    
    joint_position = robot_joint_position.detach().view(-1).numpy()
    sim_joint_position = np.zeros_like(joint_position)
    mapping_to_sim = CommonJoints.get_joints(is_right=True).get_mapping_to_sim()
    for i in range(joint_position.shape[0]):
            sim_joint_position[mapping_to_sim[i]] = joint_position[i]
    Sim_Joint_positions.append(sim_joint_position)
    joint_from_root_position = None
#print(Sim_Joint_positions)
physics_client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setGravity(0, 0, -9.8)
log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "robot_output_"+args.e+".mp4")

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
#24*10
#30*10
number=0
print("#############################")
print(len(Sim_Joint_positions))
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
                #if count%num_move==0:
                    #print("start:  "+str(int(count/num_move))+" frame")
                #time.sleep(time_step)
                '''
                infos=p.getJointStates(bodyUniqueId=ROBOT_ID,jointIndices=movable_joint_ids)
                Joint_Pos_infos=[]
                for info in infos:
                    Joint_Pos_infos.append(info[0])
                x=np.array(Joint_Pos_infos)
                y=np.array(Sim_Joint_positions[number])
                if (abs((x-y)/y)).mean()<0.05:
                   print("reach:"+str(number))
                   number+=1
                   print(number)
                time.sleep(time_step)
                if number==len(Sim_Joint_positions):
                   print("finish")
                   
                   break
                '''
            except p.error as e:
                print(e)
                pass
p.stopStateLogging(log_id)
