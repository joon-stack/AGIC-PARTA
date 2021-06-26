import cv2
import numpy as np
import os
import natsort

def prepare_data():
    p = './data/'
    input_rgb_path = []
    input_depth_path = []
    input_rgb_fname = []
    input_depth_fname = []
    lettuce_rgb_roi = []
    lettuce_depth_roi = []
    save_path = './data/ROI/'
    
    ROI_SIZE = (768, 768)

    for path, direct, files in os.walk(p):
        for f in files:
            if f.startswith('RGB_') and not path.endswith('ROI'):
                input_rgb_fname.append(f)
                input_rgb_path.append(os.path.join(path, f))
            elif f.startswith('Debth_') and not path.endswith('ROI'):
                input_depth_fname.append(f)
                input_depth_path.append(os.path.join(path, f))

    print("Got {} RGB images".format(len(input_rgb_path)))
    print("Got {} depth images".format(len(input_depth_path)))


    for fname, fname_d, savename, savename_d in zip(input_rgb_path, input_depth_path, input_rgb_fname, input_depth_fname):
        # Get RGB image and depth image simultaneously
        src = cv2.imread(fname)
        src_d = cv2.imread(fname_d, 3)
        # Take 1 channel of the depth image
        src_d = cv2.cvtColor(src_d, cv2.COLOR_BGR2GRAY)
        src_h, src_w = src.shape[:2]
        # Match template to find the most similar part with the template from the source image 
        # Get roughly cropped ROI images
        # Detect lettuce from the roughly cropped ROI images
        _, bin_d = cv2.threshold(src_d, 1000, 255, cv2.THRESH_BINARY_INV)
        bin_d = bin_d.astype('uint8')

        se = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        cv2.morphologyEx(bin_d, cv2.MORPH_OPEN, kernel=se, dst=bin_d)
        # Labeling
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_d)
        # Find lettuce ROI area
        for i in range(1, cnt):
            (x, y, w, h, area) = stats[i]
            if area < 100000:
                continue
            roi_center = (x + w // 2, y + h // 2)

            roi = src[int(roi_center[1]) - ROI_SIZE[1] // 2 : int(roi_center[1]) + ROI_SIZE[1] // 2, int(roi_center[0]) - ROI_SIZE[0] // 2 : int(roi_center[0]) + ROI_SIZE[0] // 2]
            roi_d = src_d[int(roi_center[1]) - ROI_SIZE[1] // 2 : int(roi_center[1]) + ROI_SIZE[1] // 2, int(roi_center[0]) - ROI_SIZE[0] // 2 : int(roi_center[0]) + ROI_SIZE[0] // 2]
            # roi = src[int(roi_center[1]) - ROI_SIZE[1] // 2 : int(roi_center[1]) + ROI_SIZE[1] // 2, int(roi_center[0]) - ROI_SIZE[0] // 2 : int(roi_center[0]) + ROI_SIZE[0] // 2]
            # roi_d = src_d[int(roi_center[1]) - ROI_SIZE[1] // 2 : int(roi_center[1]) + ROI_SIZE[1] // 2, int(roi_center[0]) - ROI_SIZE[0] // 2 : int(roi_center[0]) + ROI_SIZE[0] // 2]
        
        src_d = np.interp(src_d, [0, roi_d.max()], [0, 255]).astype('uint8')
        roi_d = np.interp(roi_d, [0, roi_d.max()], [0, 255]).astype('uint8')
        roi_d = roi_d.reshape((ROI_SIZE[0], ROI_SIZE[1], 1))
        src_d = src_d.reshape((src_h, src_w, 1))
    
        roi_m = np.concatenate((roi, roi_d), axis=2)
        # roi_m = roi
        roi_m = cv2.resize(roi_m, (256, 256))
        # roi_m = roi
        # roi_m.resize((256, 256, 3))
        src_m = np.concatenate((src, src_d), axis=2)
        src_m = cv2.resize(src_m, (256, 256))
        # lettuce_rgb_roi.append(roi_m) 
        lettuce_rgb_roi.append(src_m)
        # print(roi.shape)

        savename = os.path.join(save_path, savename)
        cv2.imwrite(savename, src_m)
    
        

    print("Image prepared! ")
    return lettuce_rgb_roi, input_rgb_fname




if __name__ == "__main__":
    prepare_data()       
