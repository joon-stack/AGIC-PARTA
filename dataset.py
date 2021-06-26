import os

import numpy as np
from prompt_toolkit import output
import torch
import torch.nn as nn

import natsort

# modified
import cv2
import torchvision.transforms.transforms as transforms
import copy

IMG_SIZE = (256, 256)

# Data Loader

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image, label, transform=None):
        self.transform = transform
        self.image = image
        self.label = label
        self.img_size = (256, 256)


    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        self.img_data = self.image[index][0]
        self.img_label = self.label[index]
        self.img_fname = self.image[index][1]

        return [self.img_data, self.img_label, self.img_fname]

    def custom_collate_fn(self, data):
        inputImages = []
        outputVectors = []
        fileNames = []
        h, w = self.img_size
        
        for sample in data:
            img = sample[0]
            label = sample[1]
            fname = sample[2]

            if img.ndim == 2:
                img = img[:, :, np.newaxis]
            
            inputImages.append(img.reshape((h, w, 4)))
            outputVectors.append(label)
            fileNames.append(fname)


        data = {'input': inputImages, 'label': outputVectors, 'fname': fileNames}

        if self.transform:
            data = self.transform(data)
        
        return data



class ToTensor(object):
  def __call__(self, data):
    label, input, fname = data['label'], data['input'], data['fname']
    h, w = IMG_SIZE
    input_tensor = torch.empty(len(input), 4, h, w)
    label_tensor = torch.empty(len(input), 5)
    for i in range(len(input)):
      input[i] = input[i].transpose((2, 0, 1)).astype(np.float32)
      input_tensor[i] = torch.from_numpy(input[i])
      label_tensor[i] = torch.from_numpy(label[i])

    data = {'label': label_tensor, 'input': input_tensor}

    return data
