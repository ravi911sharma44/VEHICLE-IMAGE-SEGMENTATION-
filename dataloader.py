import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image 
import matplotlib.pyplot as plt
import cv2

class CustomData(Dataset):

    def __init__(self,imgPath,maskPath):

        self.imgpath  = imgPath
        self.maskpath = maskPath
        self.allImages = sorted(os.listdir(imgPath))
        self.allMaskFolders = sorted(os.listdir(maskPath))
        self.labelMapping = []
        for maskFolder in self.allMaskFolders:
            self.labelMapping.append([int(i.split('.')[0]) for i in os.listdir(maskPath+'/'+maskFolder)])
        self.labelMapping = sorted(self.labelMapping)
        
        self.desiredAspect = 1024/576
        
        def aspectRatio(self,image,convert = None):
            width, height = image.size
            aspect = width/height
            if aspect == self.desiredAspect:
                img = Image.fromarray(image).resize((1024,576))
                if convert is not None:
                    img = img.convert(convert)
                return np.array(img)
            else:
                print('not desired ratio')
                exit(0)



    def __getitem__(self,idx):
        image = plt.imread(self.imgpath+'/'+self.allImages[idx])
        image = self.aspectRatio(image)
        target = np.zeroes([576,1024])
        maskdir = self.maskpath + '/' + self.allImages[idx][:-4]
        for maski in sorted(os.listdir(maskdir)):
            targeti = cv2.imread(maskdir+'/'+maski)
            targeti = self.aspectRatio(maski, 'L')
            target[maski == 255] = self.labelMapping.index(int(maski.split('.')[0]))
            
        return image.transpose(2,0,1).astype(np.float32)/255, target.astype(np.int64)    


    def __len__(self):
        return len(self.allImages)
        