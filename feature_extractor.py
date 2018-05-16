from __future__ import print_function, division

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torch.autograd import Variable
import time
import os
from trainer import train_model
from util import VGG_extract_transforms, VGG_dataloader, bio_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    help='Dateset directory')
parser.add_argument('--ckpt_path', type=str,
                    help='The directory where model is saved')
parser.add_argument('--class_name', type=str,
                    help='Choose which class to explain')
parser.add_argument('--save_directory', type=str,
                    help='Where to save extracted features')

args = parser.parse_args()

class Vgg19_extractor(torch.nn.Module):
    def __init__(self, ori_model):
        super(Vgg19_extractor, self).__init__()
        features = list(ori_model.features)
        self.features = nn.Sequential(*features)
        self.classifier = list(ori_model.classifier)

    def forward(self, x):
        results = {}
        x = self.features(x)
        x = x.view(x.size(0), -1)
        for ii, layer in enumerate(self.classifier):
            x = layer(x)
            if ii in {1, 6}:
                results[ii] = x
        return results

# DATA_DIR = args.data_dir
# CKPT_PATH = args.ckpt_path
# CLASS_NAME = args.class_name
# SAVE_DIRECTORY = args.save_directory

DATA_DIR = '/home/wangxiny/Bio/Model_0412_1'
CKPT_PATH = '/home/wangxiny/Bio/Breast_Tumor_With_SRAE/train_0515_1/ckpt-145.pth.tar'
CLASS_NAME = args.class_name
SAVE_DIRECTORY = args.save_directory

data_transforms = VGG_extract_transforms()
image_datasets = bio_datasets(DATA_DIR, data_transforms)
dataloaders = VGG_dataloader(image_datasets)
class_names = image_datasets['train'].classes
num_classes = len(class_names)
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}

use_gpu = torch.cuda.is_available()
model_ft = torch.load(CKPT_PATH)
model_ex = Vgg19_extractor(model_ft)
model_ex.train(False)

inputs, labels = next(iter(dataloaders['val']))
inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

temp = None
for i in range(20):
    outputs = model_ex(inputs)
    if temp is None:
        temp = outputs[6].data.cpu().numpy()
    else:
        if not np.array_equal(temp, outputs[6].data.cpu().numpy()):
            print(False)
        temp = outputs[6].data.cpu().numpy()
        print(True)
    _, preds = torch.max(outputs[6].data, 1)
    running_corrects = torch.sum(preds == labels.data)
    print(running_corrects)
