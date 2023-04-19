from torch.utils.data import random_split
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
import tqdm
import sys 
sys.path.append("..")
from tools.robot_hands_helper import CommonJoints, JointInfo


from torch.utils.data import Dataset
def split_dataset(dataset, keep=0.2):
    train_size = int(keep * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    return train_dataset, validation_dataset




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

def generate_collision_dataset(path,num):
    
    physics_client = p.connect(p.DIRECT)
    ROBOT_ID = p.loadURDF(
                os.path.join( r"/media/sdc/haosheng23/baseline/howbullet/robots/", "allegro_hand_description", "allegro_hand_description_right.urdf"),
                useFixedBase=True,flags=p.URDF_USE_SELF_COLLISION
            )
    movable_joint_ids = [i for i in range(p.getNumJoints(ROBOT_ID))
                    if i not in CommonJoints.get_joints(is_right=True).TIP]
    dataset=[]
    for _ in tqdm.tqdm(range(num)):
        random_angle=[random.uniform(angle[i]['lower_bound'],angle[i]['upper_bound']) for i in movable_joint_ids]
        for index in range(len(movable_joint_ids)):
            p.resetJointState(ROBOT_ID,movable_joint_ids[index],0)
        # reset the jointstate
        for index in range(len(movable_joint_ids)):    
            p.resetJointState(ROBOT_ID,movable_joint_ids[index],random_angle[index])
        points_pen=p.getClosestPoints(ROBOT_ID,ROBOT_ID,-0.01)
          
        if len(points_pen)>80:
            dataset.append((random_angle+[0,1]))# collision
        else:
            dataset.append((random_angle+[1,0]))# safe
    np.save(path,np.array(dataset))    


class collisionDataset(Dataset):


    def __init__(self,file,file_path=None):

        if file_path==None:
            print("使用单个文件")
            self.data=np.load(file)
        else:
            print("使用多个文件")
            file_list=[]
            for i in os.listdir(file_path):
                file_list.append(np.load(file_path+i))
            print("开始拼接")
            self.data=np.concatenate(file_list,axis=0)
            
        #data=np.load(file)
        #self.data=data
        # 1. Initialize file path or list of file names.
        
    def __getitem__(self, index):
        

        label=self.data[index][16:18]
        data=self.data[index][0:16]

        return data,label
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.data)

from torch.utils.data import IterableDataset


class LargecollisionDataset(IterableDataset):

    def __init__(self, file_path):
        super(LargecollisionDataset, self).__init__()
        
        #self.file_list = file_list # 这里设置所有待读取的文件的目录
        
        file_list=[]
        for i in os.listdir(file_path):
            file_list.append(file_path+i)
        self.file_list = file_list
    def parse_file(self):
        for file in self.file_list: # 逐个文件读取
            print("读取文件：", file)
            data=np.load(file)
            for i in data: # 逐行读取
                	# 这里可以根据具体文件格式读取，但要记得 yield 一定要返回类似于1行、1个单位的数据，可以对数据加工
                yield i[0:16],i[16:18]

    def __iter__(self):
    	# 如果 batch_size = 3,则会循环3次这个方法
        return self.parse_file()
