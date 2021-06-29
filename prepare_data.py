import cv2
import numpy as np
import os
import natsort
import copy

def augment(roi, fname, size):
    path = './data/augment/'
    if not os.path.exists(path):
        os.mkdir(path)
    augmented_images = []
    augmented_fnames = []
    for n in range(size):
        for src, f in zip(roi, fname):
            angle = 90 * (n + 1)
            h, w = src.shape[:2]
            rotation = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            dst = cv2.warpAffine(src, rotation, (w, h), borderValue=[0, 0, 0, 0])
            q = np.random.rand(1)
            if q < 0.5:
                dst = cv2.flip(dst, 1)
            else:
                dst = cv2.flip(dst, 0)
            f = 'a{}'.format(n) + f
            augmented_images.append(dst)
            augmented_fnames.append(f)
            cv2.imwrite(os.path.join(path, f), dst)
            
    return augmented_images, augmented_fnames

        

def prepare_data(mode, augmentation_size=0):
    if mode == 'train':
        p = './data'
        needAugment = True
    elif mode == 'evaluate':
        p = './evaluation'
        needAugment = False

    input_rgb_path = []
    input_depth_path = []
    input_rgb_fname = []
    input_depth_fname = []
    lettuce_rgb_roi = []
    lettuce_depth_roi = []

    save_path = os.path.join(p, 'ROI')
    
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

    rgb_idx = natsort.index_natsorted(input_rgb_fname)
    depth_idx = natsort.index_natsorted(input_depth_fname)

    input_rgb_path = natsort.order_by_index(input_rgb_path, rgb_idx)
    input_rgb_fname = natsort.order_by_index(input_rgb_fname, rgb_idx)
    input_depth_path = natsort.order_by_index(input_depth_path, depth_idx)
    input_depth_fname = natsort.order_by_index(input_depth_fname, depth_idx)

    for fname, fname_d, savename, savename_d in zip(input_rgb_path, input_depth_path, input_rgb_fname, input_depth_fname):
        # Get RGB image and depth image simultaneously
        src = cv2.imread(fname)
        src_d = cv2.imread(fname_d, 3)
        # Take 1 channel of the depth image
        src_d = cv2.cvtColor(src_d, cv2.COLOR_BGR2GRAY)
        src_h, src_w = src.shape[:2]
        _, src_d = cv2.threshold(src_d, 1000, 255, cv2.THRESH_TOZERO_INV)
        # Match template to find the most similar part with the template from the source image 
        # Get roughly cropped ROI images
        # Detect lettuce from the roughly cropped ROI images
        # _, bin_d = cv2.threshold(src_d, 1000, 255, cv2.THRESH_BINARY_INV)
        # bin_d = bin_d.astype('uint8')

        # se = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        # cv2.morphologyEx(bin_d, cv2.MORPH_OPEN, kernel=se, dst=bin_d)
        # # Labeling
        # cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_d)
        # # Find lettuce ROI area
        # for i in range(1, cnt):
        #     (x, y, w, h, area) = stats[i]
        #     if area < 100000:
        #         continue
        #     roi_center = (x + w // 2, y + h // 2)

        #     roi = src[int(roi_center[1]) - ROI_SIZE[1] // 2 : int(roi_center[1]) + ROI_SIZE[1] // 2, int(roi_center[0]) - ROI_SIZE[0] // 2 : int(roi_center[0]) + ROI_SIZE[0] // 2]
        #     roi_d = src_d[int(roi_center[1]) - ROI_SIZE[1] // 2 : int(roi_center[1]) + ROI_SIZE[1] // 2, int(roi_center[0]) - ROI_SIZE[0] // 2 : int(roi_center[0]) + ROI_SIZE[0] // 2]

        #     roi = src[int(roi_center[1]) - ROI_SIZE[1] // 2 : int(roi_center[1]) + ROI_SIZE[1] // 2, int(roi_center[0]) - ROI_SIZE[0] // 2 : int(roi_center[0]) + ROI_SIZE[0] // 2]
        #     roi_d = src_d[int(roi_center[1]) - ROI_SIZE[1] // 2 : int(roi_center[1]) + ROI_SIZE[1] // 2, int(roi_center[0]) - ROI_SIZE[0] // 2 : int(roi_center[0]) + ROI_SIZE[0] // 2]
        src_d = np.interp(src_d, [0, src_d.max()], [0, 255]).astype('uint8')
        center_h, center_w = src_h // 2, src_w // 2
        roi = src[center_h  - ROI_SIZE[1] // 2: center_h + ROI_SIZE[1] // 2, center_w - ROI_SIZE[0] // 2: center_w + ROI_SIZE[0] // 2]
        roi_d = src_d[center_h  - ROI_SIZE[1] // 2: center_h + ROI_SIZE[1] // 2, center_w - ROI_SIZE[0] // 2: center_w + ROI_SIZE[0] // 2]
        roi_d = np.interp(roi_d, [0, roi_d.max()], [0, 255]).astype('uint8')
        roi_d = roi_d.reshape((ROI_SIZE[0], ROI_SIZE[1], 1))
        
    
        roi_m = np.concatenate((roi, roi_d), axis=2)
        # roi_m = roi
        roi_m = cv2.resize(roi_m, (256, 256))

        # print(fname)
        num = int(fname.split("/")[-1].split(".")[0].split("_")[1])
        # print(num)

        # if num < 20:
        #     M = np.float32([[1, 0, -50], [0, 1, 0]])
        #     h, w = roi_m.shape[:2]
        #     roi_m = cv2.warpAffine(roi_m, M, (w, h))
        # roi_m = roi
        # roi_m.resize((256, 256, 3))
        # src_d = np.interp(src_d, [0, roi_d.max()], [0, 255]).astype('uint8')
        # src_d = src_d.reshape((src_h, src_w, 1))
        # src_m = np.concatenate((src, src_d), axis=2)
        # src_m = cv2.resize(src_m, (256, 256))
        # lettuce_rgb_roi.append(roi_m) 
        lettuce_rgb_roi.append(roi_m)
        # print(roi.shape)
        savename = os.path.join(save_path, savename)

        cv2.imwrite(savename, roi_m)
    
    lettuce_rgb_fnames = copy.copy(input_rgb_fname)

    if needAugment:
        augmented_images, augmented_fnames = augment(lettuce_rgb_roi, lettuce_rgb_fnames, augmentation_size)
        lettuce_rgb_roi.extend(augmented_images)
        lettuce_rgb_fnames.extend(augmented_fnames)

    print("Image prepared! ")

    return lettuce_rgb_roi, lettuce_rgb_fnames


if __name__ == "__main__":
    prepare_data('train')       
