import os
import torch
import pandas as pd
from pathlib import Path 
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle

class Cifar10DistribeDataset(Dataset):
    
    def __init__(self, train, divide=2, key_word='cifar-10', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.key_word = key_word
        root_path = os.path.dirname(__file__)
        data_root = Path(root_path)
        self.target = list(data_root.glob(f'*{key_word}*py'))[0] # must only one!
        if train == True:
            self.data_list = list(self.target.glob('*data_batch*'))
        else:
            self.data_list = list(self.target.glob('*test_batch*'))
        self.description = self.summary()
        self.data = []
        dim = self.description['number_dim']
        for each in self.data_list:
            sub_data =self.unpickle(each)
            name_list = [(name) for name,value in sub_data.items()]
            for lab,img in zip(sub_data[name_list[1]],sub_data[name_list[2]]):
                org_img = self.transfer_cifar_image(img)
                start_idx = 0
                if dim%divide != 0: ##########not a good way!!!! but just do now
                    start_idx = dim%divide
                else:
                    partition = int(dim/divide)
                for i in range(divide):
                    img = org_img[..., start_idx:start_idx+partition]
                    zero = np.zeros((3, 32, 16), dtype="uint8")
                    img = np.concatenate( (img,zero), axis=2)
                    lab = int(lab)
                    self.data.append((lab, img))
                    start_idx += partition
 
    def summary(self):
        summary_path = list(self.target.glob('batches.meta'))[0]
        summary = self.unpickle(summary_path)
        number_dim = int((summary[b'num_vis']/3)**0.5)
        summary['number_dim'] = number_dim
        summary['total_num'] = len(self.data_list) * summary[b'num_cases_per_batch']
        print(summary)
        return summary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lab,img = self.data[idx]
        sample = [img, lab]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def unpickle(self,file):

        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def transfer_cifar_image(self,img):
        dim = self.description['number_dim']
        _img = np.array(img, dtype="float32")
        img = _img.reshape((3,dim*dim) )
        img = img.reshape((3,dim,dim) ) #.swapaxes(0,1).swapaxes(1,2)
        # img shape should be 3,128,128
        return img