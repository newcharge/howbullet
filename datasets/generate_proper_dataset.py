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



Net = torch.laod("/media/sdc/haosheng23/baseline/howbullet/model/model_100epoch_10082223.pth")