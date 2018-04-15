from torch.autograd import Function
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

class 3dVAE(nn.Modeul):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
