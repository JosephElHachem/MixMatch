# CIFAR
import torch.nn as nn

class cifar_phi(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.conv4 = nn.Conv2d(32, 16, 3)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = x.view(-1, 9216)
        return x

class cifar_model(nn.Module):
    def __init__(self, Phi):
        super().__init__()
        self.conv = Phi
        self.fc = nn.Linear(9216, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
