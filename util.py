from __future__ import print_function, division

import torch
import torchvision
from torchvision import datasets, transforms
import os
import numpy as np
import shutil

def VGG_transforms():
    """Data preprocessing pipeline for VGG model
       #TODO: add more options for data augmentation
    """
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(1.05/1, 1/1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.15),
        transforms.ToTensor(),
        #TODO: use bio dataset mean and variance
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    return data_transforms

def bio_datasets(data_dir, data_transforms):
    """Create bio dataset
       Args:
           data_dir: training/validation data directory
                     The directory should be looklike:
                         data_dir/train/class1/class1_001.png
           data_transforms: preprocessing pipeline for your dataset
    """
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    return image_datasets

def VGG_dataloader(image_datasets):
    """Data loader for VGG model
    """
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=24,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    return dataloaders

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')
