import os
import json
import numpy as np
import pandas
import cv2
from skimage import io, transform
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import time
from os import listdir
from torchvision import transforms as T


def load_json(file_path):
    with open(file_path, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def image_convert(image):
    image = image.clone().cpu().numpy()
    image = image.transpose((1, 2, 0))
    image = (image * 255)
    return image


def mask_convert(mask):
    mask = mask.clone().cpu().detach().numpy()
    return np.squeeze(mask)


def plot_img(no_, loader, device):
    iter_ = iter(loader)
    images, masks = next(iter_)
    images = images.to(device)
    masks = masks.to(device)
    plt.figure(figsize=(20, 10))
    for idx in range(0, no_):
        image = image_convert(images[idx])
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = (image * std + mean).astype(np.float32)
        plt.subplot(2, no_, idx+1)
        plt.imshow(image)
    for idx in range(0, no_):
        mask = mask_convert(masks[idx])
        plt.subplot(2, no_, idx+no_+1)
        plt.imshow(mask, cmap='gray')
    plt.show()


class Covid19_data(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.masks, self.images = [], []
        self.transform = transform
        for file in os.listdir(os.path.join(self.path, "frames")):
            self.images.append(os.path.join(self.path, "frames", file))
        for file in os.listdir(os.path.join(self.path, "masks")):
            self.masks.append(os.path.join(self.path, "masks", file))

        self.images = sorted(self.images)
        self.masks = sorted(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        mask = cv2.imread(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        image = image / 255
        mask = mask / 255
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        image = image.transpose((2, 0, 1))
        mask = np.expand_dims(mask, axis=-1).transpose((2, 0, 1))

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        # 简单预处理
        image = T.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))(image)
        return (image, mask)


class Brain_data(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.patients = [file for file in os.listdir(path) if file not in [
            'data.csv', 'README.md']]
        self.masks, self.images = [], []
        self.transform = transform
        for patient in self.patients:
            for file in os.listdir(os.path.join(self.path, patient)):
                if 'mask' in file.split('.')[0].split('_'):
                    self.masks.append(os.path.join(self.path, patient, file))
                else:
                    self.images.append(os.path.join(self.path, patient, file))

        self.images = sorted(self.images)
        self.masks = sorted(self.masks)

    def __len__(self):
        return len(self.images)


def __getitem__(self, idx):
    image = self.images[idx]
    mask = self.masks[idx]

    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    mask = cv2.imread(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    image = image / 255
    mask = mask / 255
    if self.transform is not None:
        aug = self.transform(image=image, mask=mask)
        image = aug['image']
        mask = aug['mask']

    image = image.transpose((2, 0, 1))
    mask = np.expand_dims(mask, axis=-1).transpose((2, 0, 1))

    image = torch.from_numpy(image)
    mask = torch.from_numpy(mask)
    # 简单预处理
    image = T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))(image)
    return (image, mask)


class Breast_dataset(Dataset):

    def __init__(self, path, transforms=None):
        self.path = path
        self.transform = transforms
        self.images = []
        self.masks = []
        for img in listdir(self.path):
            if img[-8:] == "mask.png":
                self.masks.append(os.path.join(self.path, img))
            elif img[-5:] == ").png":
                self.images.append(os.path.join(self.path, img))
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        mask = cv2.imread(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        image = image / 255
        mask = mask / 255
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        image = image.transpose((2, 0, 1))
        mask = np.expand_dims(mask, axis=-1).transpose((2, 0, 1))

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        # 简单预处理
        image = T.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))(image)
        return (image, mask)

    def __len__(self):
        return len(self.images)


def save_log(name, config, train_loss, test_loss, datasets, path="Logs/", evaluations=None):
    date = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    source = {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'model': name,
        'batch_size': config["general"]["batch_size"],
        'epochs': config["general"]["epochs"],
        'if_augmentation': config["general"]["augmentation"],
        'datasets': datasets
    }
    if evaluations:
        source.update(evaluations)

    np.save(path+name+"_"+date+".log.npy", source)
    print("log saved at: "+path+date+".log.npy")


def show_config(config):
    print("Going to train these models:")
    for i in config["general"]["chosen_models"]:
        model_name = config["general"]["models"][i]
        print("\t"+model_name)
    print("On datasets:")
    for i in config["general"]["chosen_datasets"]:
        dataset_name = config["general"]["datasets"][i]
        print("\t"+dataset_name)
    print("Using evaluation:")
    for i in config["evaluation"]["chosen_methods"]:
        method_name = config["evaluation"]["methods"][i]
        print("\t"+method_name)
