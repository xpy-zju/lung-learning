from math import ceil
from statistics import mode
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, utils
import os
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model_trained1.pth')
model = model.to(device)
model.eval()

for param in model.parameters():
    param.requires_grad = False

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

def get_filename(path= 'output_frame/test',filetype ='.jpg'):
    name =[]
    final_name = []
    for root,dirs,files in os.walk(path):
        for i in files:
            if filetype in i:
                name.append(i.replace(filetype,''))
    final_name = [item + filetype for item in name]
    return final_name      #输出由有‘.jpg'后缀的文件名组成的列表

for img in get_filename():
    input = default_loader(('output_frame/test/'+img))
    input = input.to(device)
    input = input.float().unsqueeze(0)
    output = model(input)
    output = output.squeeze(0)
    output = torch.Tensor.cpu(output)

    output = output.numpy()
    print(output)

    img = cv2.imread(('output_frame/test/'+img))
    height = img.shape[0]
    width = img.shape[1]
    for num in range(ceil(3*output[0])):
        w = int(output[3*num + 1]*img.shape[0])
        h = int(output[3*num + 2]*img.shape[0])
        r = int(output[3*num + 3]*img.shape[0])
        cv2.circle(img, (w, h), r,(0, 255, 0), 2)
    cv2.namedWindow('img',0)
    cv2.imshow('img',img)
    cv2.waitKey()
# x = np.array([3,0.5,0.5,0.5,0.1,0.1,0.1,0,0,0])
# for num in range(round(x[0])):
#     print(num)

