# Imports
from cProfile import label
import copy
from turtle import forward
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets,models
import dataloader
from torch.utils.tensorboard import SummaryWriter


# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

length_label = 10

# resnet18 model
Myresnet = models.resnet18(pretrained=True)
# for param in Myresnet.parameters():
#     param.requires_grad = False

num_ftrs = Myresnet.fc.in_features #512
Myresnet.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 206),
    nn.ReLU(),
    nn.Linear(206, 10),
    nn.Sigmoid()
)

# for i, para in enumerate(Myresnet.parameters()):
#     if i < 60:
#         para.requires_grad = False
#     else:
#         para.requires_grad = True

# for name, param in Myresnet.named_parameters():
#     if param.requires_grad:
#         print("requires_grad: True ", name)
#     else:
#         print("requires_grad: False ", name)
# #test if this cause all of the requires_grad turn true
# with torch.set_grad_enabled(True):
#     for name, param in Myresnet.named_parameters():
#         if param.requires_grad:
#             print("requires_grad: True ", name)
#         else:
#             print("requires_grad: False ", name)

Myresnet = Myresnet.to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, Myresnet.parameters()), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
# exp_lr_scheduler = optim.lr_scheduler.StepLR(optim.optimizer_ft, step_size=7, gamma=0.1)
# x = torch.rand(1,3,224,224)
# x = x.to(device)
# y = Myresnet(x).to(device)
# print(y)

loss1 = nn.MSELoss()
def criterion(outputs,labels):
    # 暂时使用MSE均方损失函数，之后根据目标检测的特性优化为IOU或者其他损失函数
    cost = loss1(outputs,labels)
    return cost
# a = torch.rand(1,5)
# b = torch.rand(1,5)
# print(criterion(a,b))


class myMSEloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, outputs, labels):
        loss = 0
        for i in range(len(outputs)):
            if labels[i][0] == 0:
                loss += 10*(outputs[i][0]-labels[i][0])**2
            elif labels[i][0] == 1:
                loss += 7*(outputs[i][0]-labels[i][0])**2 + 1*(outputs[i][1]-labels[i][1])**2+ 1*(outputs[i][2]-labels[i][2])**2+ 1*(outputs[i][3]-labels[i][3])**2
            elif labels[i][0] == 2:
                loss += 4*(outputs[i][0]-labels[i][0])**2 + 1*(outputs[i][1]-labels[i][1])**2+ 1*(outputs[i][2]-labels[i][2])**2+ 1*(outputs[i][3]-labels[i][3])**2 + 1*(outputs[i][4]-labels[i][4])**2+ 1*(outputs[i][5]-labels[i][5])**2+ 1*(outputs[i][6]-labels[i][6])**2
            elif labels[i][0] == 3:
                loss += (outputs[i][0]-labels[i][0])**2 + 1*(outputs[i][1]-labels[i][1])**2+ 1*(outputs[i][2]-labels[i][2])**2+ 1*(outputs[i][3]-labels[i][3])**2 + 1*(outputs[i][4]-labels[i][4])**2+ 1*(outputs[i][5]-labels[i][5])**2+ 1*(outputs[i][6]-labels[i][6])**2+ 1*(outputs[i][7]-labels[i][7])**2+ 1*(outputs[i][8]-labels[i][8])**2+ 1*(outputs[i][9]-labels[i][9])**2
        loss = loss/len(outputs)
        return   (torch.tensor(0.0, requires_grad=True) if loss ==0  else loss)
myloss = myMSEloss()

class myMSEloss2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, outputs, labels):
        loss = 0
        for i in range(len(outputs)):
            if labels[i][0] == 0:
                loss += loss1(outputs[i][0],labels[i][0])
            elif round(3*labels[i][0].item()) == 1:
                loss += loss1(outputs[i][0:3],labels[i][0:3])
            elif round(3*labels[i][0].item()) == 2:
                loss += loss1(outputs[i][0:6],labels[i][0:6])
            elif round(3*labels[i][0].item()) == 3:
                loss += loss1(outputs[i][0:9],labels[i][0:9])
        loss = loss/len(outputs)
        return   (torch.tensor(0.0, requires_grad=True) if loss ==0  else loss)
myloss2 = myMSEloss2()

#将每张照片的label按x坐标重新排列一下 唉

#####################################
# Dataset process

Batch_size = 30
Batch_len = len(dataloader.img_path_np) / Batch_size


train_data = dataloader.mydataset_train()
trainloader = dataloader.DataLoader(train_data, Batch_size, shuffle=True)

val_data = dataloader.mydataset_val()
valloader = dataloader.DataLoader(val_data, Batch_size, shuffle=True)

My_loaders = {'train':trainloader, 'val':valloader}

print(My_loaders['train'].__len__())
print(My_loaders['val'].__len__())

#####################################

def train_model(model, loaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    least_loss = 10

    writer = SummaryWriter('log/')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for loader in loaders:
            if loader == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # running_corrects = 0 暂时没法搞corrects

            # Iterate over data.
            iter = 0
            for inputs, labels in loaders[loader]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(loader == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels.float())
                    # backward + optimize only if in training phase
                    if loader == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                Batch_loss = loss.item() * inputs.size(0)
                running_loss += Batch_loss
                # running_corrects += ?
            # if loader == 'train':
            #     scheduler.step()
            
            #Record loss in every batch
                if loader == 'train':
                    writer.add_scalar('Traning Loss for batch', Batch_loss/Batch_size, epoch*My_loaders['train'].__len__()+iter)     # loss / 30 images  -> per image loss,  No.Batch
                    iter+=1
                    percent = round( iter / My_loaders['train'].__len__() * 100, 2)
                    print('Training progress: %s [%d/%d]' % (str(percent) + '%', iter, My_loaders['train'].__len__()), end='\r')
                else:
                    writer.add_scalar('Validating Loss for batch', Batch_loss/Batch_size, epoch*My_loaders['val'].__len__()+iter)     # loss / 30 images  -> per image loss,  No.Batch
                    iter+=1
                    percent = round( iter / My_loaders['val'].__len__() * 100, 2)
                    print('Validating progress: %s [%d/%d]' % (str(percent) + '%', iter, My_loaders['val'].__len__()), end='\r')
            
            print(loaders[loader].__len__())

            epoch_loss = running_loss / loaders[loader].__len__()
            # epoch_acc = running_corrects.double() / loaders[loader].__len__()
            if loader == 'train':
                writer.add_scalar('Traning Loss for epoch', epoch_loss, epoch)
            else:
                writer.add_scalar('Validating Loss for epoch', epoch_loss, epoch)

            print('{} Loss: {:.4f} '.format(
                loader, epoch_loss))

            # deep copy the model
            if loader == 'val' and epoch_loss < least_loss:
                least_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_trained = train_model(Myresnet, My_loaders, criterion, optimizer, num_epochs=10)
torch.save(model_trained, 'model_trained.pth') 
print(Myresnet)
# test 
