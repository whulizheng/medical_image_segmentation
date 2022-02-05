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

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class TransVGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,img_size):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.middle = TransformerBlock(middle_channels,middle_channels,img_size,4)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.middle(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class Up(nn.Module):
    """Upscaling and concat"""

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x


class Model(nn.Module):
    def __init__(self, input_shape, num_classes=1, deep_supervision=False):
        super(Model, self).__init__()
        channel, height, width = input_shape
        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = Up()

        self.conv0_0 = VGGBlock(channel, nb_filter[0], nb_filter[0]) # 256 -> 128
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])# 128 -> 64
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])# 64 -> 32
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])# 32 -> 16
        self.conv4_0 = TransVGGBlock(nb_filter[3], nb_filter[4], nb_filter[4],16)# 16 -> 8
        
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.out = nn.Sigmoid()

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(self.up(x1_0, x0_0))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(self.up(x2_0, x1_0))
        x0_2 = self.conv0_2(self.up(x1_1, torch.cat([x0_0, x0_1], 1)))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(self.up(x3_0, x2_0))   
        x1_2 = self.conv1_2(self.up(x2_1, torch.cat([x1_0, x1_1], 1)))
        x0_3 = self.conv0_3(self.up(x1_2, torch.cat([x0_0, x0_1, x0_2], 1)))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(self.up(x4_0, x3_0))
        x2_2 = self.conv2_2(self.up(x3_1, torch.cat([x2_0, x2_1], 1)))
        x1_3 = self.conv1_3(self.up(x2_2, torch.cat([x1_0, x1_1, x1_2], 1)))
        x0_4 = self.conv0_4(self.up(x1_3, torch.cat([x0_0, x0_1, x0_2, x0_3], 1)))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return self.out([output1, output2, output3, output4])

        else:
            output = self.out(self.final(x0_4))
            return output

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