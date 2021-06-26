import os
import numpy as np
import torch
import torch.nn as nn
import natsort
import pandas as pd
import cv2

from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.nn import functional as F
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from dataset import *
from prepare_data import *
from copy import copy
import time

import warnings
warnings.filterwarnings('ignore')

def train(args):
    model_ckpt = './models/param.data'
    data_path = args.data_dir

    fold_size = args.fold_size
    earlyStoppingThres = 20

    
    input_images, input_fnames = prepare_data()
    input_images = np.array(input_images)
    input_fnames = np.array(input_fnames)
    test = input_images[0][:, :, :3]
    cv2.imwrite(input_fnames[0], test)
    target_df = pd.read_csv('./results/output_df.csv', index_col='Unnamed: 0')
    max_values = target_df.max().values
    min_values = target_df.min().values
    target_df = (target_df - min_values) / (max_values - min_values)
    target_df.to_csv("./results/norm_output_df.csv")
    output_labels = target_df.values

    idx = natsort.index_natsorted(input_fnames)

    input_fnames = np.array(natsort.order_by_index(input_fnames, idx))
    input_images = np.array(natsort.order_by_index(input_images, idx))

    transform = transforms.Compose([ToTensor()])

    kf = KFold(fold_size, True, random_state=1004)
    fold = 0
    fold_val_set = []
    isTrain = True if args.trainmode == 'train' else False

    print("Train mode {}".format(isTrain))

    for train_idx, val_idx in kf.split(output_labels):
        fold += 1
        
        image_train, image_val = input_images[train_idx], input_images[val_idx]
        label_train, label_val = output_labels[train_idx], output_labels[val_idx]

    
        fname_train, fname_val = input_fnames[train_idx], input_fnames[val_idx]

        image_train = list(zip(image_train, fname_train))
        image_val = list(zip(image_val, fname_val))
        num_train = len(train_idx)
        num_val = len(val_idx)
       
        # print(image_train[3][1])
        # test = (max_values - min_values) * label_train[3] + min_values
        # print(test)

        fold_val_set.append((image_val, label_val))

        if isTrain:
            earlyStoppingCount = 0

            dataset_train = CustomDataset(image_train, label_train, transform=transform)
            loader_train = DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, collate_fn=dataset_train.custom_collate_fn, num_workers=8)
            dataset_val = CustomDataset(image_val, label_val, transform=transform)
            loader_val = DataLoader(dataset_val, batch_size=num_val, shuffle=True, collate_fn=dataset_val.custom_collate_fn, num_workers=8)

            model = models.resnet152(pretrained=True)
            # for param in model.parameters():
            #     param.requires_grad = False
            model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(2048, 5, bias=True)
            model.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.1, training=m.training))

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # device = torch.device('cpu')
            print('Current device:', device)
            
            model.to(device)

            criterion = nn.MSELoss().to(device)

            optim = torch.optim.Adam(model.parameters(), lr=0.0001)

            best_epoch = 0
            val_loss_save = np.array(np.inf)

            for epoch in range(args.epochs):
                model.train()
                train_loss = []
                
                for batch, data in enumerate(loader_train, 1):
                    label = data['label'].to(device)
                    input = data['input'].to(device)
                    output = model(input)
                    
                    optim.zero_grad()

                    loss = criterion(output, label)
                    loss.backward()

                    optim.step()

                    train_loss += [loss.item()]

                with torch.no_grad():
                    model.eval()
                    val_loss = []
                    for batch, data in enumerate(loader_val, 1):
                        label_val = data['label'].to(device)
                        input_val = data['input'].to(device)
                        output_val = model(input_val)
                        loss = criterion(output_val, label_val)
                        val_loss += [loss.item()] 

                    param = list(model.parameters())

                    val_loss_tmp = loss.item()

                    earlyStoppingCount += 1
                    if val_loss_tmp < val_loss_save:
                        earlyStoppingCount = 0
                        best_epoch = epoch
                        val_loss_save = copy(val_loss_tmp)
                        torch.save(model.state_dict(), './models/param{}.data'.format(fold))

                        print(".......model updated (epoch = ", epoch+1, ")")
                    print("epoch: %04d / %04d | train loss: %.5f | validation loss: %.5f" %
                    (epoch+1, args.epochs, np.mean(train_loss), np.mean(val_loss) ))
                    if earlyStoppingCount > earlyStoppingThres:
                        print("Early stopped")
                        break

            print("Model with the best validation accuracy is saved.")
            print("Best epoch: ", best_epoch)
            print("Best validation loss: ", val_loss_save)
            print("Done.")

    return fold_val_set



                
            
            

        



