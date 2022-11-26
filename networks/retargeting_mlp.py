import torch.nn as nn

from tools.common import rescale_robot_joint_position


class RetargetingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            #nn.Linear(55, 256), nn.Tanh(),
            # change 55---> 45

            #nn.Linear(45, 256), nn.Tanh(),
            nn.Linear(63,256),nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 16), nn.Tanh()
        )

    def forward(self, x):
        tanh = self.model(x)
        output = rescale_robot_joint_position(tanh)
        return output
