import torch
import os
from PIL import Image
import numpy as np

class FaceDataset(torch.utils.data.Dataset):
  def __init__(self,train=True,type='normal'):
    super().__init__()
    if(train):
      self.data = np.genfromtxt(os.path.join('dataset','train.csv'),delimiter=',',dtype=None,encoding='utf-8')
    else:
      self.data = np.genfromtxt(os.path.join('dataset','test.csv'),delimiter=',',dtype=None,encoding='utf-8')

    self.type=type

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    img_name = self.data[index][0]
    X = Image.open(os.path.join('dataset','happy_images','{}.jpg'.format(img_name)))
    label = self.data[index][1]

    if(self.type=='normal'):
      if(label=='NOT smile'):
        y = 0
      else:
        y = 1
    else:
      if(label=='positive smile'):
        y = 1
      else:
        y = 0

    return X,y


if(__name__=='__main__'):
  d = FaceDataset()