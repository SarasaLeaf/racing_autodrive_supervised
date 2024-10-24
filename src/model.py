# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self, num_keys=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(64 * 24 * 24, 100)  # 224x224入力の場合
        self.fc2 = nn.Linear(100, num_keys)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 24, H/2, W/2]
        x = F.relu(self.conv2(x))  # [B, 48, H/4, W/4]
        x = F.relu(self.conv3(x))  # [B, 64, H/8, W/8]
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # バイナリ分類なのでシグモイド
        return x

class ResNetModel(nn.Module):
    def __init__(self, num_keys=5, pretrained=True):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_keys)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.sigmoid(x)  # バイナリ分類なのでシグモイド
        return x
