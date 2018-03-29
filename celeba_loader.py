import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


mypath = 'C:\dataset\celebA\img_align_celeba\img_align_celeba\\'
attripath = 'C:\dataset\celebA\list_attr_celeba.txt'
attributelist = ['Big_Lips','Big_Nose','Eyeglasses', 'High_Cheekbones','Male','Mouth_Slightly_Open','Mustache'
                ,'Narrow_Eyes','No_Beard','Pale_Skin','Pointy_Nose','Smiling','Young']
#print(mypath)
#onlyfiles = [f for f in os.listdir(mypath)]
#print(onlyfiles)

class celebaldr(object):
    def __init__(self,path,batch_size):
         self.path = path
         self.bn   = batch_size
         self.cnt  = 0
         self.getlist()
         self.loadattribute(self.list)
    def getlist(self):
         self.list = [f for f in os.listdir(self.path)]
         self.lens = len(self.list)
         print("len : %s" %self.lens)
    def getbn(self,crop = True):
         if self.cnt+self.bn >= self.lens:
          self.cnt = 0
          return 0 , []
         else :
          
          batch_img = torch.FloatTensor()
          one_hot_label = []
          trans = transforms.ToTensor()
          for i in range(self.bn):
           file = self.path + self.list[i+self.cnt]
           im = Image.open(file)
           one_hot_label.append(self.file_attri[self.list[i+self.cnt]])
           if crop :
            im = im.crop((60, 80, 60 + 64, 80 + 80))
           im=im.resize((64,64))
           batch_img = torch.cat((batch_img,trans(im).unsqueeze(0)))
         self.cnt += self.bn
         one_hot_label = np.array(one_hot_label).reshape(self.bn,-1)
         #print(one_hot_label)
         return batch_img,one_hot_label
    def loadattribute(self,name_list):
        f = open(attripath,'r')
        nums = f.readline()
        attribute = f.readline()
        attribute = attribute.split()
        #print(attribute)
        dic_attri = {attribute[i] : i for i in range(0,len(attribute) ) }
        #print(dic_attri)
        self.key_attri = [dic_attri[var] for var in attributelist ]
        i = 0
        file_attri = {}
        while True :
            buffer = f.readline()
            if not buffer:
                break
            buffer = buffer.split()
            file_attri[buffer[0]] = [ 1 if int(buffer[int(var)+1]) == 1 else 0 for var in self.key_attri ]
        self.file_attri = file_attri
        f.close()


           

#cel = celebaldr(mypath,2)
#cel.getbn()
#cel.getbn()
     
