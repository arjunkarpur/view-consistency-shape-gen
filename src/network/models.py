from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

class AE_3D(nn.Module):
    def __init__(self, resolution=20, embed_space=64):
        super(AE_3D, self).__init__()

        # Encoder
        self.conv1 = nn.Conv3d(1, 96, 11, stride=4)
        self.maxpool1 = nn.MaxPool3d(3, stride=2)
        self.conv2 = nn.Conv3d(96, 256, 5, padding=2, groups=2)
        self.maxpool2 = nn.MaxPool3d(3, stride=2)
        self.conv3 = nn.Conv3d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv3d(384, 384, 3, padding=1, groups=2)
        self.conv5 = nn.Conv3d(384, 256, 3, padding=1, groups=2)
        self.maxpool5 = nn.MaxPool3d(3, stride=2)
        self.fc6 = nn.Linear(256*6*6*6, 4096)
        self.fc7_embed = nn.Linear(4096, embed_space)

        # Decoder
        #TODO
        return;

    def forward(self, x):
        # TODO
        #embed = self._encode(x)
        #y = self._decode(embed)
        y = x
        return y

    def _encode(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool5(F.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc6(x)))
        x = self.fc7_embed(x)
        return x

    def _decode(self, x):
        #TODO
        pass
