import torch
import random
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import Dataset, DataLoader 
import os
import PIL
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

class custum_data_loader(Dataset):
    
    def __init__(self, imagepath, labelpath, labelmapping = None):
        
        self.imagepath=imagepath
        self.labelpath=labelpath
        if labelmapping is None:
            self.labelMapping = set()
            for labelI in tqdm(os.listdir(labelpath)):
                self.labelMapping.update(set([int(i.split('.')[0]) for i in os.listdir(labelpath+'/'+labelI)]))
            self.labelMapping = sorted(list(self.labelMapping)) 
        else:
            self.labelMapping = labelmapping       
        self.allimages =  sorted(os.listdir(self.imagepath))
        self.imageshape=[180,320]
        self.desiredaspect=self.imageshape[1]/self.imageshape[0]

        
    def changeimagesize(self, image,convert = None):

        aspect=image.shape[1]/image.shape[0]
        if aspect==self.desiredaspect:
            image=Image.fromarray(image).resize((self.imageshape[1],self.imageshape[0]))
            if convert is not None:
                image = image.convert(convert)
            
            return self.quantize(np.array(image))*255
        else:
            print('Image is not of desired aspect ratio!!')
            exit(0)
    
    def quantize(self,label):
        return (label != 0).astype(np.int64)  

    def __getitem__(self, item):
        
        image = plt.imread(self.imagepath + '/' + self.allimages[item])
        image=self.changeimagesize(image)
        target=np.zeros([image.shape[0],image.shape[1]])
        labelfolder=self.labelpath+'/'+self.allimages[item][:-4]
        
        for labelI in sorted(os.listdir(labelfolder)):
            targetI = cv2.imread(labelfolder+'/'+labelI)[:,:,0]
            targetI = self.changeimagesize(targetI, 'L')
            target[targetI== 255]=int(labelI.split('.')[0])
        
        return image.transpose(2,0,1).astype(np.float32)/255, target.astype(np.int64)
            
    def __len__(self):
        return len(self.allimages)


class UNet(nn.Module):
    
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block
    
    def forward(self, X):
        contracting_11_out = self.contracting_11(X) # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
        contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, 16, 16]
        middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
        expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(expansive_42_out) # [-1, num_classes, 256, 256]
        return output_out

def seed(seed_value):



    torch.cuda.manual_seed_all(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    model = UNet(num_classes=20)    
    use_cuda = True
    seed(0)

    if use_cuda:
        model = model.cuda()
    epochs = 10
    lr = 0.01
    dataset = custum_data_loader('E:/AUTONOMOUS VEHICLE IMAGE SEGMENTATION intern/week 4&5/train','E:/AUTONOMOUS VEHICLE IMAGE SEGMENTATION intern/week 4&5/uniqueMasks')
    data_loader = DataLoader(dataset,num_workers=4,shuffle=True,batch_size=2)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    step_losses = []
    epoch_losses = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for X, Y in tqdm(data_loader, total=len(data_loader), leave=False):
            if use_cuda:
                X, Y = X.cuda(),Y.cuda()
            optimizer.zero_grad()
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_losses.append(loss.item())
        epoch_losses.append(epoch_loss/len(data_loader))
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, epoch_loss/len(data_loader)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(step_losses)
    axes[1].plot(epoch_losses)    

    model_name = "U-Net.pth"
    torch.save(model.state_dict(), model_name)

if __name__ == '__main__':
    main()
