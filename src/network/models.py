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
        nn.init.normal(self.conv1_3d.weight, mean=0, std=0.01)
        self.prelu_en_1 = nn.PReLU()
        self.bn1_3d = nn.BatchNorm3d(96)

        self.conv2_3d = nn.Conv3d(96, 256, 5)
        nn.init.normal(self.conv2_3d.weight, mean=0, std=0.01)
        self.prelu_en_2 = nn.PReLU()
        self.bn2_3d = nn.BatchNorm3d(256)

        self.conv3_3d = nn.Conv3d(256, 384, 3)
        nn.init.normal(self.conv3_3d.weight, mean=0, std=0.01)
        self.prelu_en_3 = nn.PReLU()
        self.bn3_3d = nn.BatchNorm3d(384)

        self.conv4_3d = nn.Conv3d(384, 256, 3)
        nn.init.normal(self.conv4_3d.weight, mean=0, std=0.01)
        self.prelu_en_4 = nn.PReLU()
        self.bn4_3d = nn.BatchNorm3d(256)

        self.fc5_embed = nn.Linear(256*6*6*6, embed_space)
        nn.init.normal(self.fc5_embed.weight, mean=0, std=0.01)

        # Setup decoder
        self.fc6_reconstruct = nn.Linear(embed_space, 216)
        nn.init.normal(self.fc6_reconstruct.weight, mean=0, std=0.01)

        self.deconv4_3d = nn.ConvTranspose3d(1, 256, 3)
        nn.init.normal(self.deconv4_3d.weight, mean=0, std=0.01)
        self.prelu_de_4 = nn.PReLU()

        self.deconv3_3d = nn.ConvTranspose3d(256, 384, 3)
        nn.init.normal(self.deconv3_3d.weight, mean=0, std=0.01)
        self.prelu_de_3 = nn.PReLU()

        self.deconv2_3d = nn.ConvTranspose3d(384, 256, 5)
        nn.init.normal(self.deconv2_3d.weight, mean=0, std=0.01)
        self.prelu_de_2 = nn.PReLU()

        self.deconv1_3d = nn.ConvTranspose3d(256, 96, 7)
        nn.init.normal(self.deconv1_3d.weight, mean=0, std=0.01)
        self.prelu_de_1 = nn.PReLU()

        self.deconv0_3d = nn.ConvTranspose3d(96, 1, 1)
        nn.init.normal(self.deconv0_3d.weight, mean=0, std=0.01)
        return;

    def forward(self, x):
        x = x.view(x.size(0), 1, self.res, self.res, self.res)
        x = self._encode_norm(x)
        x = self._decode(x)
        return x

    def _encode_norm(self, x):
        x = self.bn1_3d(self.prelu_en_1(self.conv1_3d(x)))
        x = self.bn2_3d(self.prelu_en_2(self.conv2_3d(x)))
        x = self.bn3_3d(self.prelu_en_2(self.conv3_3d(x)))
        x = self.bn4_3d(self.prelu_en_2(self.conv4_3d(x)))
        x = x.view(x.size(0), -1)
        x = self.fc5_embed(x)
        return x

    def _decode(self, x):
        x = self.fc6_reconstruct(x)
        x = x.view(x.size(0), 1, 6, 6, 6)
        x = self.prelu_de_4(self.deconv4_3d(x))
        x = self.prelu_de_3(self.deconv3_3d(x))
        x = self.prelu_de_2(self.deconv2_3d(x))
        x = self.prelu_de_1(self.deconv1_3d(x))
        x = F.sigmoid(self.deconv0_3d(x))
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(4))
        return x
