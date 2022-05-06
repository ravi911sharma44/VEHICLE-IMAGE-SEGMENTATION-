
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

class LoadingData(Dataset):

    def __init__(self, imagepath, labelpath, labelmapping = None):
        
        self.imagepath=imagepath
        self.labelpath=labelpath
        if labelmapping is None:
            self.labelMapping = set()
            for labelI in os.listdir(labelpath):
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()       
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,16,3,padding=1),                            # batch x 16 x H x W
                        nn.ReLU(),
                        nn.BatchNorm2d(16),
                        nn.Conv2d(16,32,3,padding=1),                           # batch x 32 x H x W
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,64,3,padding=1),                           # batch x 32 x H x W
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2,2)                                       # batch x 64 x H/2 x W/2
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),                          # batch x 64 x H/2 x W/2
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(128,256,3,padding=1),                         # batch x 64 x H/4 x W/4
                        nn.ReLU()
        )
        
        self.layer3 = nn.Sequential(
                        nn.ConvTranspose2d(256,128,3,2,1,1),                    # batch x 128 x H/2 x W/2
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,64,3,1,1),                       # batch x 64 x H/2 x W/2
                        nn.ReLU(),
                        nn.BatchNorm2d(64)
        )
        self.layer4 = nn.Sequential(
                        nn.ConvTranspose2d(64,16,3,1,1),                        # batch x 16 x H/2 x W/2
                        nn.ReLU(),
                        nn.BatchNorm2d(16),
                        nn.ConvTranspose2d(16,20,3,2,1,1),                       # batch x 1 x H x W 
                        nn.Sigmoid()
        )
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)
        return 2*x-1       

def train(model,use_cuda,train_data,optimizer,epoch):

    model.train()
    outputs= []
    targets= []
    train_loss = 0

    for batch_idx ,(img,target) in enumerate(train_data):

        if use_cuda:
            img,target = img.cuda(),target.cuda()

        optimizer.zero_grad()

        output = model(img)
        loss = F.mse_loss(output,target)
    
             
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        outputs.append(output[-1].cpu().detach().numpy())
        targets.append(target[-1].cpu().detach().numpy())
    i = outputs[-1]
    j = targets[-1]
    outimg = i.transpose(1,-1,0)
    tarimg = j.transpose(1,-1,0)

    f, plots = plt.subplots(1,2)
    plots[0].imshow((outimg+1)/2)
    plots[1].imshow((tarimg+1)/2)
    os.makedirs('training_plots3',exist_ok=True)
    plt.savefig("training_plots3/{}.png".format(epoch))
    train_loss = train_loss/len(train_data) 
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))



def seed(seed_value):



    torch.cuda.manual_seed_all(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    use_cuda = True

    seed(0)
    train_data = DataLoader(LoadingData('E:/AUTONOMOUS VEHICLE IMAGE SEGMENTATION intern/week 2/10k/train'),num_workers=4,shuffle=True,batch_size=2)

    model = Net()

    

    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    for epoch in tqdm(range(1, 10 + 1)):
        train(model, use_cuda, train_data, optimizer, epoch) 
   
    torch.save(model, "autoencoders_3.pt")  

if __name__ == '__main__':
    main()


