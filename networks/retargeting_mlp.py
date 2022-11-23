import torch.nn as nn

from tools.common import rescale_robot_joint_position


class RetargetingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(63, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 20), nn.Tanh()  # 16 for angles and 4 for rotation from canonical of root
        )

    def forward(self, x):
        tanh = self.model(x)
        joints, rot = tanh[:, :16], tanh[:, 16:]
        return rescale_robot_joint_position(joints), rot
