import torch.nn as nn

from tools.common import rescale_robot_joint_position
from tools.fk_helper import robot_joint_angle_to_coordinate
from tools.torch_canonical.trans import transform_to_canonical


class RetargetingMLP(nn.Module):
    def __init__(self, chains):
        super().__init__()
        self.chains = chains
        self.model = nn.Sequential(
            nn.Linear(63, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 16), nn.Tanh()
        )

    def forward(self, x):
        tanh = self.model(x)
        robot_joint_angles = rescale_robot_joint_position(tanh)
        joint_from_base_coordinates = robot_joint_angle_to_coordinate(
            chains=self.chains, robot_joint_angles=robot_joint_angles
        )
        joints_position_from_canonical, base2canonical = transform_to_canonical(
            joint_from_base_coordinates, is_human=False
        )
        return robot_joint_angles, base2canonical, joints_position_from_canonical
