import torch
from torch import nn
import torch.nn.functional as F
from torch import channel_shuffle, nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from statistics import mean
from torch.autograd import Variable
import time
from PIL import Image
import random
from skimage import io, transform
from collections import OrderedDict
import numpy as np

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils

from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from matplotlib import pyplot as plt
from skimage import io, transform
from torch.nn import init
from torchvision.models.resnet import BasicBlock, ResNet


# Returns 2D convolutional layer with space-preserving padding
def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
    if transposed:
        layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                                   stride=stride, padding=1, output_padding=1, dilation=dilation, bias=bias)
        # Bilinear interpolation init 用双线性插值法初始化反卷积核
        w = torch.Tensor(kernel_size, kernel_size)
        centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
        for y in range(kernel_size):
            for x in range(kernel_size):
                w[y, x] = (1 - abs((x - centre) / stride)) * \
                    (1 - abs((y - centre) / stride))
        layer.weight.data.copy_(
            w.div(in_planes).repeat(in_planes, out_planes, 1, 1))
    else:
        padding = (kernel_size + 2 * (dilation - 1)) // 2
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                          stride=stride, padding=padding, dilation=dilation, bias=bias)
    if bias:
        init.constant(layer.bias, 0)
    return layer


# Returns 2D batch normalisation layer
def bn(planes):
    layer = nn.BatchNorm2d(planes)
    # Use mean 0, standard deviation 1 init
    init.constant(layer.weight, 1)
    init.constant(layer.bias, 0)
    return layer


class FeatureResNet(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 4, 6, 3], 1000)  # 特征提取用resnet

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.bn1(x1)
        x = self.relu(x)
        x2 = self.maxpool(x)
        x = self.layer1(x2)
        x3 = self.layer2(x)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5


class Model(nn.Module):
    def __init__(self, input_shape, pretrained_net):
        super().__init__()
        channel, height, width = input_shape
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.trans = TransformerBlock(512,512,8,2)
        self.conv5 = conv(512, 256, stride=2, transposed=True) #256
        self.bn5 = bn(256)
        self.conv6 = conv(256, 128, stride=2, transposed=True) #128
        self.bn6 = bn(128)
        self.conv7 = conv(128, 64, stride=2, transposed=True) #64
        self.bn7 = bn(64)
        self.conv8 = conv(64, 64, stride=2, transposed=True) #32
        self.bn8 = bn(64)
        self.conv9 = conv(64, 32, stride=2, transposed=True)# 16
        self.bn9 = bn(32)
        self.conv10 = conv(32, 1, kernel_size=7)
        self.out = nn.Sigmoid()
        init.constant(self.conv10.weight, 0)  # Zero init

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.pretrained_net(x)
        x5 = self.trans(x5)
        x = self.relu(self.bn5(self.conv5(x5)))
        x = self.relu(self.bn6(self.conv6(x + x4)))
        x = self.relu(self.bn7(self.conv7(x + x3)))
        x = self.relu(self.bn8(self.conv8(x + x2)))
        x = self.relu(self.bn9(self.conv9(x + x1)))
        x = self.conv10(x)
        x = self.out(x)
        return x


def dice_coef_metric(pred, label):
    intersection = 2.0 * (pred * label).sum()
    union = pred.sum() + label.sum()
    if pred.sum() == 0 and label.sum() == 0:
        return 1.
    return intersection / union


def dice_coef_loss(pred, label):
    smooth = 1.0
    intersection = 2.0 * (pred * label).sum() + smooth
    union = pred.sum() + label.sum() + smooth
    return 1 - (intersection / union)


class Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        dice_loss = dice_coef_loss(inputs, targets)
        bce_loss = nn.BCELoss()(inputs, targets)
        return dice_loss + bce_loss


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, emb_size, patch_size: int = 8, img_size: int = 256):
        assert emb_size == patch_size**2
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size,
                      kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.positions = nn.Parameter(torch.randn(
            (img_size // patch_size)**2, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, emb_size, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = channels

        self.keys = nn.Linear(emb_size, emb_size*self.num_heads)
        self.queries = nn.Linear(emb_size, emb_size*self.num_heads)
        self.values = nn.Linear(emb_size, emb_size*self.num_heads)

        self.multihead_attn = nn.MultiheadAttention(
            emb_size*self.num_heads, 1, batch_first=True, dropout=dropout)  # 多头部分自己处理，所以这里只是调用torch的attention

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)

        att = self.multihead_attn(queries, keys, values, need_weights=False)[0]

        out = rearrange(att, "b n (h d) -> b h n d", h=self.num_heads)
        return out


class SpecialResidualAdd(nn.Module):
    def __init__(self, fn, channels):
        super().__init__()
        self.channels = channels
        self.fn = fn

    def forward(self, x, **kwargs):
        tmp = torch.unsqueeze(x,0)
        for i in range(self.channels-1):
            tmp = torch.cat((tmp,torch.unsqueeze(x,0)),0)
        tmp = rearrange(tmp, "h b n d -> b h n d")
        x = self.fn(x, **kwargs)
        x += tmp
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self,
                 channels,
                 emb_size,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            SpecialResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(channels,emb_size, **kwargs),
                nn.Dropout(drop_p)
            ), channels),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, patch_size):
        super(TransformerBlock, self).__init__()
        self.emb_size = (patch_size**2)
        self.patch_size = patch_size
        self.img_size = img_size
        self.batchnorm = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.pe = PatchEmbedding(
            in_channels=in_channels, out_channels=out_channels, emb_size=self.emb_size, patch_size=patch_size, img_size=img_size)
        self.encoder = TransformerEncoder(
            channels=out_channels, emb_size=self.emb_size)
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        )
    def forward(self, x):
        out = self.pe(x)
        out = self.encoder(out)
        out = rearrange(out, "b h (n o) (s p) -> b h (n s) (o p)",
                        n=int(self.img_size/self.patch_size), s=self.patch_size)
        out = self.deconv(out)
        out = self.batchnorm(out)
        out = self.relu(out)
        return out