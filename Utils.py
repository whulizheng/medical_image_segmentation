import os
import json
import numpy as np
from skimage import io, transform
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms as T


def load_json(file_path):
    with open(file_path, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict

def image_convert(image):
    image = image.clone().cpu().numpy()
    image = image.transpose((1,2,0))
    image = (image * 255)
    return image

def mask_convert(mask):
    mask = mask.clone().cpu().detach().numpy()
    return np.squeeze(mask)

def plot_img(no_,loader,device):
    iter_ = iter(loader)
    images,masks = next(iter_)
    images = images.to(device)
    masks = masks.to(device)
    plt.figure(figsize=(20,10))
    for idx in range(0,no_):
         image = image_convert(images[idx])
         mean = [0.485, 0.456, 0.406]
         std = [0.229, 0.224, 0.225] 
         image = (image * std + mean).astype(np.float32)
         plt.subplot(2,no_,idx+1)
         plt.imshow(image)
    for idx in range(0,no_):
         mask = mask_convert(masks[idx])
         plt.subplot(2,no_,idx+no_+1)
         plt.imshow(mask,cmap='gray')
    plt.show()

class Brain_data(Dataset):
    def __init__(self,path,transform=None):
        self.path = path
        self.patients = [file for file in os.listdir(path) if file not in ['data.csv','README.md']]
        self.masks,self.images = [],[]
        self.transform = transform
        for patient in self.patients:
            for file in os.listdir(os.path.join(self.path,patient)):
                if 'mask' in file.split('.')[0].split('_'):
                    self.masks.append(os.path.join(self.path,patient,file))
                else: 
                    self.images.append(os.path.join(self.path,patient,file)) 
          
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = io.imread(image)
        image = image / 255
        mask = io.imread(mask)
        mask = mask / 255
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        
        image = image.transpose((2, 0, 1))
        mask = np.expand_dims(mask,axis=-1).transpose((2, 0, 1))

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        # 简单预处理
        image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        return (image,mask)