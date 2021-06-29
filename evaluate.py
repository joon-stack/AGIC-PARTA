import os
import numpy as np
import torch
import torch.nn as nn
import natsort
import pandas as pd
import argparse
import json

from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.nn import functional as F
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from dataset import *
from prepare_data import *
from copy import copy
import time

def evaluate(args):
    model_name = args.model_name
    answer = args.answer
    result_path = './evaluation/eval.csv'
    input_images, input_fnames = prepare_data('evaluate')

    target_df = pd.read_csv('./results/output_df.csv', index_col='Unnamed: 0')
    max_values = target_df.max().values
    min_values = target_df.min().values
    img_idx = ['Image'+ s.split("_")[1].split(".")[0] for s in input_fnames]
    
    if answer:
        target_df = (target_df - min_values) / (max_values - min_values)
        target_df = target_df.loc[img_idx, :]
        output_labels = target_df.values
    else:
        output_labels = np.zeros((len(input_images), 5))
    
    input_images = list(zip(input_images, input_fnames))
   
    model_ckpt = './models/{}.data'.format(model_name)

    print("Current model:", model_ckpt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device:', device)

    model = models.resnet152(pretrained=False)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, 5, bias=True)

    transform = transforms.Compose([ToTensor()])

    if model_ckpt:
        state_dict = torch.load(model_ckpt)
        state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
        model.load_state_dict(state_dict)
    
    
    with torch.no_grad():
        model.eval()
        model = model.cuda()
        num_input = len(input_fnames)

        
        dataset_val = CustomDataset(input_images, output_labels, transform=transform)
        loader_val = DataLoader(dataset_val, batch_size=num_input, shuffle=False, collate_fn=dataset_val.custom_collate_fn, num_workers=8)

        for batch, data in enumerate(loader_val, 1):
            val_label = data['label'].to(device)
            val_input = data['input'].to(device)
            pred_label = model(val_input)
        
        result_y = (max_values - min_values) * pred_label.cpu().numpy() + min_values
        result_x = (max_values - min_values) * val_label.cpu().numpy() + min_values

        for y in result_y:
            for a in range(len(y)):
                if y[a] < 0:
                    y[a] = 0
        if answer:
            final_output = np.concatenate([result_x, result_y], axis=1)
            final_output = pd.DataFrame(final_output, index=img_idx, columns=[f'label_{_}' for _ in target_df.columns] + [f'pred_{_}' for _ in target_df.columns])

            squared_error = np.zeros(5)
            squared_target = np.zeros(5)

            for n, cname in enumerate(target_df.columns):
                squared_error[n] += np.sum(np.power(final_output['label_{}'.format(cname)] - final_output['pred_{}'.format(cname)], 2))
                squared_target[n] += np.sum(np.power(final_output['label_{}'.format(cname)], 2))

            nmse = squared_error / squared_target
            print(nmse)
            print("NMSE: {:.4f}".format(np.sum(nmse)))

            final_output.to_csv(result_path)

        columns = ["FreshWeightShoot", "DryWeightShoot", "Height", "Diameter", "LeafArea"]
        columns_output = ["RGBImage", "DebthInformation"]
        columns_output.extend(columns)
        final_output = pd.DataFrame(result_y, index=img_idx, columns=[f'{_}' for _ in columns])
        final_output['RGBImage'] = input_fnames
        depth_input_fnames = [fname.replace('RGB', 'Debth') for fname in input_fnames]
        final_output['DebthInformation'] = depth_input_fnames
        final_output = final_output[columns_output]
        result = final_output.to_json(orient="index")
        parsed = json.loads(result)
        parsed = {'Measurements': parsed}
        json.dumps(parsed, indent='\t')
        file_path = os.path.join('./evaluation', (model_name + '.json'))
        with open(file_path, 'w') as outfile:
            json.dump(parsed, outfile, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="AGIC-PART A", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", default='best', type=str, dest="model_name") 
    parser.add_argument("--answer", default=1, type=int, dest="answer")

    return parser.parse_args()

def main():
    args = parse_args()
    evaluate(args)

if __name__ == '__main__':
    main()