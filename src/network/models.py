from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo

class AE_3D(nn.Module):
    def __init__(self, resolution=20, embed_space=64):
        super(AE_3D, self).__init__()

        # Save input vars
        self.res = resolution
        self.embed_space = embed_space

        # Setup encoder
        self.conv1_3d = nn.Conv3d(1, 96, 7)
        self.conv2_3d = nn.Conv3d(96, 256, 5)
        self.conv3_3d = nn.Conv3d(256, 384, 3)
        self.conv4_3d = nn.Conv3d(384, 256, 3)
        self.fc5_embed = nn.Linear(256*6*6*6, embed_space)

        # Setup decoder
        self.fc6_reconstruct = nn.Linear(embed_space, 216)
        self.deconv4_3d = nn.ConvTranspose3d(1, 256, 3)
        self.deconv3_3d = nn.ConvTranspose3d(256, 384, 3)
        self.deconv2_3d = nn.ConvTranspose3d(384, 256, 5)
        self.deconv1_3d = nn.ConvTranspose3d(256, 96, 7)
        self.deconv0_3d = nn.ConvTranspose3d(96, 1, 1)
        return;

    def forward(self, x):
        x = x.view(x.size(0), 1, self.res, self.res, self.res)
        x = self._encode(x)
        x = self._decode(x)
        return x

    def _encode(self, x):
        x = F.leaky_relu(self.conv1_3d(x))
        x = F.leaky_relu(self.conv2_3d(x))
        x = F.leaky_relu(self.conv3_3d(x))
        x = F.leaky_relu(self.conv4_3d(x))
        x = x.view(x.size(0), -1)
        x = self.fc5_embed(x)
        return x

    def _decode(self, x):
        x = self.fc6_reconstruct(x)
        x = x.view(x.size(0), 1, 6, 6, 6)
        x = F.leaky_relu(self.deconv4_3d(x))
        x = F.leaky_relu(self.deconv3_3d(x))
        x = F.leaky_relu(self.deconv2_3d(x))
        x = F.leaky_relu(self.deconv1_3d(x))
        x = F.sigmoid(self.deconv0_3d(x))
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(4))
        return x
