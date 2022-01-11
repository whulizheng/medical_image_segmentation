import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
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


class name_net(nn.Module):
    def __init__(self, input_shape):
        super(name_net, self).__init__()

        channel, height, width = input_shape
        self.conv1 = Conv2d(channel, 24, kernel_size=(1, 1), bias=True)
        self.a1 = TransformerBlock(24)
        self.a2 = TransformerBlock(24)
        self.a3 = TransformerBlock(24)
        self.a4 = TransformerBlock(24)
        self.a5 = TransformerBlock(24)
        self.a6 = TransformerBlock(24)
        self.a7 = TransformerBlock(24)
        self.a8 = TransformerBlock(24)
        self.a9 = TransformerBlock(24)
        
        self.conv2 = Conv2d(24, 1, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.a1(out)
        out = self.a2(out)
        out = self.a3(out)
        out = self.a4(out)
        out = self.a5(out)
        out = self.a6(out)
        out = self.a7(out)
        out = self.a8(out)
        out = self.a9(out)
        out = self.conv2(out)
        return out


class StackEncoder(nn.Module):
    def __init__(self, channel1, channel2, kernel_size=(3, 3), padding=1):
        super(StackEncoder, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ConvBlock(channel1, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),
        )

    def forward(self, x):
        big_out = self.block(x)
        poolout = self.maxpool(big_out)
        return big_out, poolout


class StackDecoder(nn.Module):
    def __init__(self, big_channel, channel1, channel2, kernel_size=(3, 3), padding=1):
        super(StackDecoder, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channel1+big_channel, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.upsample(x, size=(height, width), mode='bilinear')
        # combining channels of  input from encoder and upsampling input
        x = torch.cat([x, down_tensor], 1)
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
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

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        dice_loss = dice_coef_loss(inputs, targets)
        bce_loss = nn.BCELoss()(inputs, targets)
        return dice_loss + bce_loss
    
    

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU * 100

    
class DiceScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceScore, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        intersection = 2.0 * (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        if inputs.sum() == 0 and targets.sum() == 0:
            return 1.
        return intersection / union
        


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 8, emb_size: int = 3):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # 用卷积层代替线性层->性能提升
            nn.Conv2d(in_channels, emb_size,
                      kernel_size=patch_size, stride=patch_size, padding=1),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # 在输入前添加cls标记
        x = torch.cat([cls_tokens, x], dim=1)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 将查询、键和值融合到一个矩阵中
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # 分割num_heads中的键、查询和值
        qkv = rearrange(
            self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # 最后一个轴上求和
        # batch, num_heads, query_len, key_len
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # 在第三个轴上求和
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


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


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerBlock(nn.Module):
    def __init__(self,in_channels):
        super(TransformerBlock,self).__init__()
        self.pe = PatchEmbedding(
            in_channels=in_channels, emb_size=in_channels)
        self.encoder = TransformerEncoderBlock(emb_size=in_channels)
        self.deconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        )

    def forward(self, x):
        B, C, H, W = x.size()
        out = self.pe(x)
        out = self.encoder(out)
        out = out.permute(0, 2, 1)
        out = F.upsample(out, size=W*H)
        out = out.view(
            B, C, H, W)
        out = self.deconv(out)
        return out
