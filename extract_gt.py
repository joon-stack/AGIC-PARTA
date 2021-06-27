import json
import pandas as pd
import numpy as np
import natsort

FILE_DIRECTORY = './data/groundtruth/GroundTruth.json'
with open(FILE_DIRECTORY) as data_file:    
    JSON_data = json.load(data_file)


shoot_fw = []
shoot_dw = []
height = []
diameter = []
leafarea = []
variety = []

keys = np.array(list(JSON_data['Measurements'].keys()))
keys = natsort.natsorted(keys)
for _ in keys:
    shoot_fw.append(JSON_data['Measurements'][_]['FreshWeightShoot'])
    shoot_dw.append(JSON_data['Measurements'][_]['DryWeightShoot'])
    height.append(JSON_data['Measurements'][_]['Height'])
    diameter.append(JSON_data['Measurements'][_]['Diameter'])
    leafarea.append(JSON_data['Measurements'][_]['LeafArea'])
    v = JSON_data['Measurements'][_]['Variety']
    if v == 'Aphylion':
        t = 0
    elif v == 'Lugano':
        t = 1
    elif v == 'Satine':
        t = 2
    elif v == 'Salanova':
        t = 3
    variety.append(t)

output_df = pd.DataFrame(np.array([shoot_fw, shoot_dw, height, diameter, leafarea]).T, index=keys, columns=['shoot_fw', 'shoot_dw', 'height', 'diameter', 'leafarea'])
variety_df = pd.DataFrame(np.array([variety]).T, index=keys, columns=[ 'variety'])
output_df.to_csv('./results/output_df.csv')
variety_df.to_csv('./results/variety_df.csv')