import os
import numpy as np
import torch
import torch.nn as nn
import natsort
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.nn import functional as F
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from dataset import *
from prepare_data import *
from copy import copy
import time

def validate(fold_val_set):
    fold_size = len(fold_val_set)
    target_df = pd.read_csv('./results/output_df.csv', index_col='Unnamed: 0')
    max_values = target_df.max().values
    min_values = target_df.min().values
    target_df = (target_df - min_values) / (max_values - min_values)
    output_labels = target_df.values

    result_path = './results/pred_df.csv'

    transform = transforms.Compose([ToTensor()])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device:', device)

    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 5, bias=True)

    for n in range(fold_size):
        model_ckpt = './models/param{}.data'.format(n+1)
        state_dict = torch.load(model_ckpt)
        state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
        model.load_state_dict(state_dict)
        image_best, label_best = fold_val_set[n]

 
        with torch.no_grad():
            model.eval()
            model = model.cuda()
            num_best = len(image_best)
            fname_best = [img[1] for img in image_best]
            dataset_best = CustomDataset(image_best, label_best, transform=transform)
            loader_best = DataLoader(dataset_best, batch_size=num_best, shuffle=False, collate_fn=dataset_best.custom_collate_fn, num_workers=8)

            for batch, data in enumerate(loader_best, 1):
                label_best = data['label'].to(device)
                input_best = data['input'].to(device)
                output_best = model(input_best)

            result_x = (max_values - min_values) * label_best.cpu().numpy() + min_values
            result_y = (max_values - min_values) * output_best.cpu().numpy() + min_values

            final_output = np.concatenate([result_x, result_y], axis=1)
            final_output = pd.DataFrame(final_output, index=fname_best, columns=[f'label_{_}' for _ in target_df.columns] + [f'pred_{_}' for _ in target_df.columns])

            squared_error = np.zeros(5)
            squared_target = np.zeros(5)

            for n, cname in enumerate(target_df.columns):
                squared_error[n] += np.sum(np.power(final_output['label_{}'.format(cname)] - final_output['pred_{}'.format(cname)], 2))
                squared_target[n] += np.sum(np.power(final_output['label_{}'.format(cname)], 2))

            nmse = squared_error / squared_target
            print(nmse)
            print("NMSE: {:.4f}".format(np.sum(nmse)))

            final_output.to_csv(result_path)