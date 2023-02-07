from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv


train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data,mode):
        self.data = data
        self.mode=mode
        self.mean=train_mean
        self.std=train_std
        self._transform= tv.transforms.Compose([tv.transforms.ToPILImage(),tv.transforms.ToTensor(),tv.transforms.Normalize(self.mean,self.std)])
        self._label_transform=tv.transforms.Compose([tv.transforms.ToTensor()])
        # TODO Andere Transform für Valitation?

    def __getitem__(self, idx):
        img = imread(self.data.iloc[idx,0])
        label = torch.tensor([self.data.iloc[idx,1],self.data.iloc[idx,2]])
        img_rgb = gray2rgb(img)
        img_rgb = self._transform(img_rgb)
        return img_rgb,label


    def __len__(self):
        return self.data.shape[0]

