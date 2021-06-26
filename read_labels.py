import os
import json
import numpy as np
import pandas as pd
import natsort

def read_labels():
    dir = './data/groundtruth/GroundTruth.json'

    with open(dir) as f:
        data = json.load(f)

    image_index = np.array(list(data['Measurements'].keys()))
    image_index = natsort.natsorted(image_index)

    shoot_fw = []
    shoot_dw = []
    height = []
    diameter = []
    leafarea = []
    variety = []
    for _ in image_index:
        shoot_fw.append(data['Measurements'][_]['FreshWeightShoot'])
        shoot_dw.append(data['Measurements'][_]['DryWeightShoot'])
        height.append(data['Measurements'][_]['Height'])
        diameter.append(data['Measurements'][_]['Diameter'])
        leafarea.append(data['Measurements'][_]['LeafArea'])
        variety.append(data['Measurements'][_]['Variety'])

    output_df = pd.DataFrame(np.array([shoot_fw, shoot_dw, height, diameter, leafarea]).T, index=image_index, columns=['shoot_fw', 'shoot_dw', 'height', 'diameter', 'leafarea'])
    output_df.to_csv('./results/output_df.csv')

def main():
    read_labels()

if __name__ == '__main__':
    main()