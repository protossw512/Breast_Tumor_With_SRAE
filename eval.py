from __future__ import print_function, division

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models
import time
import os
from trainer import train_model
from util import VGG_transforms, VGG_dataloader, bio_datasets

# Parse arguments
#parser = argparse.ArgumentParser()
#parser.add_argument('--data_dir', type=str, required=True, default='/home/wangxiny/Bio/Breast_Tumor_With_SRAE/ckpt-90.pth',
#                    help='Dateset directory')
#parser.add_argument('--model_name', type=str, required=True, default='VGG19_bn',
#                    help='which base model to use')
#parser.add_argument('--ckpt_path', type=str, required=True,
#                    help='which checkpoint to load')
#
#args = parser.parse_args()
#
#DATA_DIR = args.data_dir
#MODEL_NAME = args.model_name
#CKPT_PATH = args.ckpt_path

DATA_DIR = '/home/wangxiny/Bio/Val_0412_CE'
MODEL_NAME = 'VGG19_bn'
CKPT_PATH = '/home/wangxiny/Bio/Breast_Tumor_With_SRAE/ckpt-45.pth.tar'

data_transforms = VGG_transforms()
image_datasets = bio_datasets(DATA_DIR, data_transforms)
dataloaders = VGG_dataloader(image_datasets)
class_names = image_datasets['val'].classes
num_classes = len(class_names)
dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}

use_gpu = torch.cuda.is_available()
model_ft = torch.load(CKPT_PATH)
# model_ft = models.vgg19_bn(pretrained=False)
# num_ftrs = model_ft.classifier[6].in_features
# convert all the layers to list and remove the last one
# features = list(model_ft.classifier.children())[:-1]
# Add the last layer based on the num of classes in our dataset
# features.extend([nn.Linear(num_ftrs, num_classes)])
# convert it into container and add it to our model class.
# model_ft.classifier = nn.Sequential(*features)

model_ft = model_ft.cuda()

# model_ft.load_state_dict(torch.load(CKPT_PATH))
model_ft.train(False)

running_corrects = 0
for data in dataloaders['val']:
    inputs, labels = data
    if use_gpu:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    # forward
    outputs = model_ft(inputs)
    _, preds = torch.max(outputs.data, 1)
    running_corrects += torch.sum(preds == labels.data)
pred_acc = running_corrects / dataset_sizes['val']

print('Acc: {:.4f}'.format(pred_acc))
