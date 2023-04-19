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
# set the environment


joint_from_root_position = None

angle=[ {'name': b'link_0.0', 'lower_bound': -0.47, 'upper_bound': 0.47},       
  {'name': b'link_1.0', 'lower_bound': -0.196, 'upper_bound': 1.61} ,     
  {'name': b'link_2.0', 'lower_bound': -0.174, 'upper_bound': 1.709} ,    
  {'name': b'link_3.0', 'lower_bound': -0.227, 'upper_bound': 1.618} ,   
  {'name': b'link_3.0_tip', 'lower_bound': 0.0, 'upper_bound': -1.0} ,
  {'name': b'link_4.0', 'lower_bound': -0.47, 'upper_bound': 0.47} ,      
  {'name': b'link_5.0', 'lower_bound': -0.196, 'upper_bound': 1.61}  ,    
  {'name': b'link_6.0', 'lower_bound': -0.174, 'upper_bound': 1.709}   ,  
  {'name': b'link_7.0', 'lower_bound': -0.227, 'upper_bound': 1.618} ,    
  {'name': b'link_7.0_tip', 'lower_bound': 0.0, 'upper_bound': -1.0} ,
  {'name': b'link_8.0', 'lower_bound': -0.47, 'upper_bound': 0.47} ,     
  {'name': b'link_9.0', 'lower_bound': -0.196, 'upper_bound': 1.61} ,    
  {'name': b'link_10.0', 'lower_bound': -0.174, 'upper_bound': 1.709} , 
  {'name': b'link_11.0', 'lower_bound': -0.227, 'upper_bound': 1.618} , 
  {'name': b'link_11.0_tip', 'lower_bound': 0.0, 'upper_bound': -1.0},
  {'name': b'link_12.0', 'lower_bound': 0.263, 'upper_bound': 1.396} ,  
  {'name': b'link_13.0', 'lower_bound': -0.105, 'upper_bound': 1.163} , 
  {'name': b'link_14.0', 'lower_bound': -0.189, 'upper_bound': 1.644}  ,
  {'name': b'link_15.0', 'lower_bound': -0.162, 'upper_bound': 1.719}  ,
  {'name': b'link_15.0_tip', 'lower_bound': 0.0, 'upper_bound': -1.0}]
points_pen_set=[]

for _ in range(100):
    physics_client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
#p.setGravity(0, 0, -9.8)

    PLANE_ID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
    ROBOT_ID = p.loadURDF(
                os.path.join( "robots", "allegro_hand_description", "allegro_hand_description_right.urdf"),
                useFixedBase=True,flags=p.URDF_USE_SELF_COLLISION
            )
            # set the center of mass frame (load URDF sets base line frame)
    movable_joint_ids = [i for i in range(p.getNumJoints(ROBOT_ID))
                    if i not in CommonJoints.get_joints(is_right=True).TIP]


    p.resetBasePositionAndOrientation(ROBOT_ID, [0, 0, 0.5], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetDebugVisualizerCamera(
            cameraDistance=1,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0.5]
        )
    random_angle=[random.uniform(angle[i]['lower_bound'],angle[i]['upper_bound']) for i in movable_joint_ids]
        # let the state be zero
    for index in range(len(movable_joint_ids)):
            p.resetJointState(ROBOT_ID,movable_joint_ids[index],0)
        # reset the jointstate
    for index in range(len(movable_joint_ids)):    
            p.resetJointState(ROBOT_ID,movable_joint_ids[index],random_angle[index])
        
    points_pen=p.getClosestPoints(ROBOT_ID,ROBOT_ID,-0.01)
    # points_pen_set.append(points_pen)  
    # get the 3d position of vec
    vec_point=[]
    for point_info in points_pen:
            vec_point.append(point_info[5])
    print(len(vec_point))
    while True:
        p.addUserDebugPoints(np.array(vec_point),np.array([(255,0,0)for i in vec_point]),pointSize=5)
        if not p.isConnected():
                    break

