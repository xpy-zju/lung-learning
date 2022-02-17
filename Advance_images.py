# Train ResNet50

#some code adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Lambda
import random
import numbers
from torch.utils.tensorboard import SummaryWriter
import cv2
import os
import numpy as np
import pickle

root = '/home/xiangpy/lab/2022_win/'                                    #Work dir             #Video
img_dir = os.path.join(root, 'output_frame/001')                      #Images

#ls output_frame: 001(Original) 002(Horizontal) 003(Vertical) 004(Rotate90) 005(Rotate270) 006(Color)

def get_files(root_dir):                           
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths


def get_filename(path= 'output_frame/001',filetype ='.jpg'):
    name =[]
    final_name = []
    for root,dirs,files in os.walk(path):
        for i in files:
            if filetype in i:
                name.append(i.replace(filetype,''))#生成不带‘.json’后缀的文件名组成的列表
    final_name = [item + filetype for item in name]#生成‘.json’后缀的文件名组成的列表
    return final_name      #输出由有‘.json'后缀的文件名组成的列表
# print(get_filename(path, filetype))


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

# Flip the image
class HorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):

        return img.transpose(Image.FLIP_LEFT_RIGHT)

class VerticalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):

        return img.transpose(Image.FLIP_TOP_BOTTOM)

# Rotate the image
class Rotation90(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):

        return img.transpose(Image.ROTATE_90)

class Rotation270(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):

        return img.transpose(Image.ROTATE_270)


class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img):
        seed = self.count // 1 #sequence_length
        random.seed(seed)
        self.count += 1
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img, brightness_factor)
        img_ = TF.adjust_contrast(img_, contrast_factor)
        img_ = TF.adjust_saturation(img_, saturation_factor)
        img_ = TF.adjust_hue(img_, hue_factor)

        return img_

# Horizontal FLip 002
Horizontal_flip = transforms.Compose([
        transforms.Resize((224, 224)),
        HorizontalFlip()
        ])

Vertical_flip = transforms.Compose([
        transforms.Resize((224, 224)), 
        VerticalFlip()
        ])

Rotate_90 = transforms.Compose([
        transforms.Resize((224, 224)), 
        Rotation90()
        ])

Rotate_270 = transforms.Compose([
        transforms.Resize((224, 224)), 
        Rotation270(),
        ])

Color_J = transforms.Compose([
        transforms.Resize((224, 224)), 
        ColorJitter(0.5,0.5,0.5,0.5)
        ])

img_file_names = get_filename()  # Get the file path and name of the original version
#print(img_file_names[3])

for i in range(len(img_file_names)):    # len(img_file_names) = 4401
    img = pil_loader('output_frame/001/' + img_file_names[i])

    Two = Horizontal_flip(img)
    Two.save("./output_frame/002/"+img_file_names[i])

    Three = Vertical_flip(img)
    Three.save("./output_frame/003/"+img_file_names[i])

    Four = Rotate_90(img)
    Four.save("./output_frame/004/"+img_file_names[i])

    Five = Rotate_270(img)
    Five.save("./output_frame/005/"+img_file_names[i])

    Six = Color_J(img)
    Six.save("./output_frame/006/"+img_file_names[i])

