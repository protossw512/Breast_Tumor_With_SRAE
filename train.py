from __future__ import print_function, division

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import time
import os
from trainer import train_model
from util import VGG_transforms, VGG_dataloader, bio_datasets


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    help='Dateset directory')
parser.add_argument('--model_name', type=str, default='VGG19_bn',
                    help='which base model to use')
parser.add_argument('--ckpt_save_path', type=str,
                    help='a directory to save checkpoint file')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate to use during training')
parser.add_argument('--learning_rate_decay_rate', type=float, default=0.1,
                    help='learning rate exponential decay rate')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--max_epoch', type=int, default=10,
                    help='number of epoches to train')
parser.add_argument('--decay_per_epoch', type=int, default=25,
                    help='learning rate decay after each epoch')

args = parser.parse_args()

DATA_DIR = args.data_dir
# TODO: make a list of models to choose from by MODEL_NAME
MODEL_NAME = args.model_name
CKPT_SAVE_PATH = args.ckpt_save_path
LEARNING_RATE = args.learning_rate
LEARNING_RATE_DECAY = args.learning_rate_decay_rate
WEIGHT_DECAY = args.weight_decay
MAX_EPOCH = args.max_epoch
DECAY_PER_EPOCH = args.decay_per_epoch

# Import dataset and datasetloaders
# data_dir = "/home/wangxiny/Bio/Model_0412_1"
data_transforms = VGG_transforms()
image_datasets = bio_datasets(DATA_DIR, data_transforms)
dataloaders = VGG_dataloader(image_datasets)
class_names = image_datasets['train'].classes
num_classes = len(class_names)
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# Load VGG19_bn base model for transfer learning
use_gpu = torch.cuda.is_available()
model_ft = models.vgg19_bn(pretrained=True)
num_ftrs = model_ft.classifier[6].in_features
# convert all the layers to list and remove the last one
features = list(model_ft.classifier.children())[:-1]
## Add the last layer based on the num of classes in our dataset
features.extend([nn.Linear(num_ftrs, num_classes)])
## convert it into container and add it to our model class.
model_ft.classifier = nn.Sequential(*features)

model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE,
                         momentum=0.9, weight_decay=WEIGHT_DECAY)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=DECAY_PER_EPOCH,
                                       gamma=LEARNING_RATE_DECAY)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       dataloaders, dataset_sizes, CKPT_SAVE_PATH, use_gpu,
                       num_epochs=MAX_EPOCH)

