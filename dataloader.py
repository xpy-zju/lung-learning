# 自定义数据读取
# read the img and the label
# 1)得到一个长 list1 : 里面是每张图片的路径
# 2)另外一个长list2: 里面是每张图片对应的标签（整数），顺序要和list1对应。
# 3)把这两个list切分出来一部分作为验证集

from cProfile import label
from tkinter.ttk import LabeledScale
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2 as cv
import random

label_file = pd.read_csv('Label/sorted.csv')

# print(df.info())
# print(label_file.head())
label_file = label_file.sample(frac=1)

img_path = label_file['path']
img_path_np = img_path.values

np.random.shuffle(img_path_np)
# print(img_path_np)
#记得随机处理视频图像数据,给图片文件随机重命名就可以了
#将数据集7/3 分为训练和验证
img_path_np_train = img_path_np[:int(0.7*len(img_path_np))]
img_path_np_val = img_path_np[int(0.7*len(img_path_np)):]
# print((img_path_np_train.shape))
# print((img_path_np_val.shape))

# np.save( "file_train.npy" ,img_path_np_train )
# np.save( "file_test.npy" ,img_path_np_val )

label_data_np = label_file[['whole_num','x1','y1','r1','x2','y2','r2','x3','y3','r3']].values
# print((label_data_np.shape))

label_data_np_train = label_data_np[0:int(0.7*len(img_path_np)),:]
label_data_np_val = label_data_np[int(0.7*len(img_path_np)):len(img_path_np),:]

# print((label_data_np_train.shape))
# print((label_data_np_val.shape))

# print(label_data_np_train[1])
# print(img_path_np_train[1])


#上面成功把训练集和测试集分为两部分,并得到各自对应的np列表,图片路径和标签一一对应

preprocess = transforms.Compose([
    transforms.Resize([224,224]),
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(
    # mean=[0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225])
])


def default_loader(path):
    img_pil =  Image.open(path)
    # img_pil = img_pil.resize((224,224))
    img_tensor = preprocess(img_pil)
    return img_tensor

# 告诉它你所有数据的长度，每次给你返回一个shuffle过的index,以这个方式遍历数据集，通过 __getitem__(self, index)返回一组你要的（input,target）
class mydataset_train(Dataset):
    def __init__(self, loader=default_loader):
        self.images = img_path_np_train
        self.target = label_data_np_train
        self.loader = loader
    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img, target
    def __len__(self):
        return len(self.images)

class mydataset_val(Dataset):
    def __init__(self, loader=default_loader):
        self.images = img_path_np_val
        self.target = label_data_np_val
        self.loader = loader
    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img, target
    def __len__(self):
        return len(self.images)

# train_data0 = mydataset_train()
# trainloader0 = DataLoader(train_data0, batch_size=4, shuffle=True)

# for epoch in range(1):
#     for i,data in enumerate(trainloader0):
#         inputs, Labels = data
#         print(inputs.size())

# img = Image.open('1.jpg')
# w, h = img.size
# resize = transforms.Resize([224,244])
# img = resize(img)
# img.save('2.jpg')
# resize2 = transforms.Resize([h, w])
# img = resize2(img)
# img.save('3.jpg')
