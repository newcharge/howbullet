import torch.nn as nn

from tools.common import rescale_robot_joint_position


class c_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
        
            nn.Linear(16,256),nn.Tanh(),
            nn.Linear(256,256),nn.Tanh(),
            #nn.Linear(256,64),nn.Sigmoid(),
            nn.Linear(256, 16), nn.Tanh(),
            nn.Linear(16, 2), nn.Tanh(),
            #nn.Linear(16,64),nn.ELU(),
            #nn.Linear(64,64),nn.ELU(),
            #nn.Linear(64,2),nn.ELU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.model(x)
        return output
    