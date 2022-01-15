import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import sqrt

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


class Model(nn.Module):
    def __init__(self, input_shape,patch_size = 8):
        super(Model, self).__init__()

        channel, height, width = input_shape
        self.name = "Transformer_CNN_Unet_mix_p1"
        self.down1 = StackEncoder(channel,12,kernel_size=(3,3))  # in: 256, out: 128
        self.down2 = StackEncoder(12,24,kernel_size=(3,3))  # in: 128, out: 64
        self.down3 = StackEncoder(24,46,kernel_size=(3,3))  # in: 64, out: 32
        self.down4 = StackEncoder(46,64,kernel_size=(3,3))  # in: 32, out: 16
        self.down5 = TransformerStackEncoder(64,128,16,4) # in: 16, out: 8
        
        self.center = ConvBlock(128,128,kernel_size=(3,3),padding=1)# in: 8, out: 8
        
        self.up5 = TransformerStackDecoder(128,128,64,16,4) # in: 8, out: 16
        self.up4 = StackDecoder(64,64,46,kernel_size=(3,3))# in: 16, out: 32
        self.up3 = StackDecoder(46,46,24,kernel_size=(3,3))# in: 32, out: 64
        self.up2 = StackDecoder(24,24,12,kernel_size=(3,3))# in: 64, out: 128
        self.up1 = StackDecoder(12,12,12,kernel_size=(3,3))# in: 128, out: 256
        self.conv = OutConv(12,1)

    def forward(self, x):
        down1,out = self.down1(x)  
        down2,out = self.down2(out)  
        down3,out = self.down3(out)
        down4,out = self.down4(out)
        down5,out = self.down5(out)
        
        
        out = self.center(out)
        
        up5 = self.up5(out,down5)
        up4 = self.up4(up5,down4)
        up3 = self.up3(up4,down3)
        up2 = self.up2(up3,down2)
        up1 = self.up1(up2,down1)
        
        out = self.conv(up1)

        return out

class TransformerStackEncoder(nn.Module):
    def __init__(self, channel1, channel2,img_size, patch_size):
        super(TransformerStackEncoder, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = TransformerBlock(channel1,channel2,img_size, patch_size)
    def forward(self, x):
        big_out = self.block(x)
        poolout = self.maxpool(big_out)
        return big_out, poolout


class TransformerStackDecoder(nn.Module):
    def __init__(self, big_channel, channel1, channel2, img_size, patch_size):
        super(TransformerStackDecoder, self).__init__()
        self.block = nn.Sequential(
            TransformerBlock(channel1+big_channel,channel2,img_size, patch_size),
            TransformerBlock(channel2,channel2,img_size, patch_size),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.upsample(x, size=(height, width), mode='bilinear')
        # combining channels of  input from encoder and upsampling input
        x = torch.cat([x, down_tensor], 1)
        x = self.block(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)


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
            emb_size*self.num_heads, 8, batch_first=True, dropout=dropout)  # 多头部分自己处理，所以这里只是调用torch的attention

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
        emb_size = (patch_size**2)
        self.patch_size = patch_size
        self.img_size = img_size
        self.batchnorm = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.pe = PatchEmbedding(
            in_channels=in_channels, out_channels=out_channels, emb_size=emb_size, patch_size=patch_size, img_size=img_size)
        self.encoder = TransformerEncoder(
            channels=out_channels, emb_size=emb_size)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.pe(x)
        out = self.encoder(out)
        out = rearrange(out, "b h (n o) (s p) -> b h (n s) (o p)",
                        n=int(self.img_size/self.patch_size), s=self.patch_size)
        out = self.batchnorm(out)
        out = self.relu(out)
        return out


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
