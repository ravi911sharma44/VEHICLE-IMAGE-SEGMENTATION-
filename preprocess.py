import torch
from torch.utils.data import Dataset , DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class custumData(Dataset): # using torch custom dataloader to do the masking task so that we can get advantage of parallel computation of torch dataloader , or we can do the same task one by one by using the code mentioned at the bottom of page

    def __init__(self,path):
        self.path = path
        self.AllLabelPath = sorted(os.listdir(path)) # here we used os library to convert all the files name in a given directory in to a sorted list

    def BreakIntoSub(self,label,outputdir): #here label is defined as a masked label which is in gray scale image
        os.makedirs(outputdir,exist_ok=True) #if thr given directory is there then this fucnction will use it or it will create one automatically
        uniqueLabels = np.unique(label) # find out all the unique pixel values inthe given ray sclae image
        for uni in uniqueLabels: # we iterate over all the unique pixel values and convert them into masks of pixel values  0 or 1 
            plt.imsave(outputdir+'/'+str(uni)+'.png',(label == uni).astype(np.uint8)*255,cmap = 'gray') #each pixel with that particular unique value will be converted to value 1(black) and the rest of pixels will be 0(white)


    def __getitem__(self,idx):
        label = (plt.imread(self.path + "/" + self.AllLabelPath[idx][:-3] + 'png')*255).astype(np.uint8) #getting the original masked label to convert it into 8 diffrent target images
        self.BreakIntoSub(label,'uniqueMasks'+"/" + self.AllLabelPath[idx][:-4]) #applying the function corresponding function to make above mentioned targets
        return 'uniqueMasks'+"/" + self.AllLabelPath[idx][:-4]
    def __len__(self):
        return len(self.AllLabelPath) #length of the object 



if __name__ == "__main__":

    customDataLoaderObject = DataLoader(
        custumData("labels/sem_seg/masks/{}rain".format('t')),batch_size=10,
        num_workers=4,
        shuffle=True
    ) # load the data

    for targets in tqdm(customDataLoaderObject): # iterate over the batches 

        print(len(targets))
        
          

#def BreakIntoSub(label,outputdir):
#    os.makedirs(outputdir,exist_ok=True)
#    uniqueLabels = np.unique(label)
#    for uni in uniqueLabels:
#        plt.imsave(outputdir+'/'+ str(uni)+'.png',(label == uni).astype(np.uint8)*255,cmap = 'gray')


#labelList  = sorted(os.listdir("labels/sem_seg/masks/{}rain".format('t')))     

#for labelpath in tqdm(labelList):
    #label = (plt.imread("labels/sem_seg/masks/{}rain".format('t') + "/" + labelpath[:-3] + 'png')*255).astype(np.uint8)
    #BreakIntoSub(label,'uniqueMasks'+"/" + labelpath[:-4])
