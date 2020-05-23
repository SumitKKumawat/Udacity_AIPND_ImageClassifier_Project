#importing dependencies here

#import Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)

#import pytorch
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models

#import time and json
import time
import json

#import numpy, argparse, sys and OS
import numpy as np
import argparse, sys
import os
os.environ['QT_QPA_PLATFORM']='offscreen'


#Data Directory for all the flower images
data_dir = 'flowers'

#deviding data into training, testing and validation
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
#normalizing data for train, test and validaiton
data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'valid': transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                   }

#directory path for train, test and validation data images
#required for below load dataset
directories = {'train': train_dir, 
               'valid': valid_dir, 
               'test' : test_dir}

# TODO: Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(directories[x], transform=data_transforms[x])
                  for x in ['train', 'valid', 'test']}