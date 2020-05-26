from pycocotools.coco import COCO
import numpy as np
import os
import cv2
from imgaug.augmentables.heatmaps import HeatmapsOnImage
import pandas as pd
import random
random.seed(1920)

from torch.utils.data import Dataset

from src.dataset.utils import get_keypoint_mask
from src.dataset.augmentations import get_augmentations


class PoseDataset(Dataset):
    def __init__(self, type, dataset_root, pafmap_joints, keypoint_points,
                 input_shape=(288, 512, 3), augment=False, downscale=4):
        self.type = type
        self.dataset_root = dataset_root
        self.input_shape = input_shape
        self.augment = augment

        if self.augment:
            self.seq = get_augmentations()

        self.pafmap_joints = pafmap_joints
        self.keypoints_points = keypoint_points

        self.downscale = downscale
        
        self.keypoint_mask = get_keypoint_mask()

        self.dataset = self.load_coco_ann_files()

    def load_coco_ann_files(self):
        if self.type == 'train':
            datasets = [(os.path.join(self.dataset_root, 'train2014'),
                         COCO(os.path.join(self.dataset_root,
                                           'annotations_trainval2014',
                                           'person_keypoints_train2014.json'))),
                        (os.path.join(self.dataset_root, 'train2017'),
                         COCO(os.path.join(self.dataset_root,
                                           'annotations_trainval2017',
                                           'person_keypoints_train2017.json')))]
        else:
            datasets = [(os.path.join(self.dataset_root, 'val2014'),
                         COCO(os.path.join(self.dataset_root,
                                           'annotations_trainval2014',
                                           'person_keypoints_val2014.json'))),
                        (os.path.join(self.dataset_root, 'val2017'),
                         COCO(os.path.join(self.dataset_root,
                                           'annotations_trainval2017',
                                           'person_keypoints_val2017.json')))]

        dict_list = []
        for dataset_path, dataset in datasets:
            img_ids = dataset.getImgIds()

            for idx in img_ids:
                try:
                    img = dataset.loadImgs([idx])[0]
                    ann_ids = dataset.getAnnIds([idx])
                    anns = dataset.loadAnns(ann_ids)

                    if [ann['keypoints'] for ann in anns] and not all([ann['keypoints'] == [0]*51 for ann in anns]):
                        dict_list.append({'path': os.path.join(dataset_path, img["file_name"]),
                                          'keypoints': [ann['keypoints'] for ann in anns]})
                except:
                    print(f'Skipped: {idx}')

        final_dataset = pd.DataFrame.from_dict(dict_list)

        return final_dataset

    def apply_keypoint_mask(self, img_keypoints, input_shape, kp_size=30):
        keypoint_masks = [np.zeros((input_shape[0] + kp_size, input_shape[1] + kp_size, 1), dtype=np.float32)
                          for _ in range(len(self.keypoints_points))]

        for kp in img_keypoints:
            for i, k in enumerate(self.keypoints_points):

                if kp[k * 3 + 2] != 0:
                    keypoint_masks[i][kp[k * 3 + 1]:kp[k * 3 + 1] + kp_size, kp[k * 3]:kp[k * 3] + kp_size, 0] = \
                        np.maximum(keypoint_masks[i][kp[k * 3 + 1]:kp[k * 3 + 1] + kp_size, kp[k * 3]:kp[k * 3] + kp_size, 0], self.keypoint_mask)

        keypoint_masks = keypoint_masks if np.max(keypoint_masks) < 0.5 else keypoint_masks / np.max(keypoint_masks)

        return np.squeeze(np.array(keypoint_masks)).transpose((1, 2, 0))[kp_size // 2:-kp_size // 2, kp_size // 2:-kp_size // 2, :]

    def apply_segmap_mask(self, img_keypoints, input_shape, thickness=12):
        seg_masks = [np.zeros((input_shape[0], input_shape[1], 1), dtype=np.float32)
                    for _ in range(len(self.pafmap_joints))]

        for kp in img_keypoints:
            for i, (kp0, kp1) in enumerate(self.pafmap_joints):

                if kp[kp0 * 3 + 2] != 0 and kp[kp1 * 3 + 2] != 0:
                    cv2.line(seg_masks[i],
                             (kp[kp0 * 3], kp[kp0 * 3 + 1]),
                             (kp[kp1 * 3], kp[kp1 * 3 + 1]),
                             255,
                             thickness)

        seg_masks = np.array(seg_masks)

        seg_masks = seg_masks if np.max(seg_masks) < 0.5 else seg_masks / np.max(seg_masks)

        return np.squeeze(seg_masks).transpose((1, 2, 0))

    def apply_pafmap_mask(self, joints, width=8):
        pass

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]

        img = cv2.imread(row['path'])
        img_keypoints = row['keypoints']

        pafmap_mask = self.apply_segmap_mask(img_keypoints, img.shape)

        keypoint_mask = self.apply_keypoint_mask(img_keypoints, img.shape)

        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        pafmap_mask = cv2.resize(pafmap_mask, (self.input_shape[1], self.input_shape[0]))
        keypoint_mask = cv2.resize(keypoint_mask, (self.input_shape[1], self.input_shape[0]))

        if self.augment:
            seq_det = self.seq.to_deterministic()

            pafmap_mask = HeatmapsOnImage(pafmap_mask, shape=img.shape, min_value=0.0, max_value=1.0)

            keypoint_mask = HeatmapsOnImage(keypoint_mask, shape=img.shape, min_value=0.0, max_value=1.0)

            img = seq_det.augment_image(img)
            pafmap_mask = seq_det.augment_heatmaps(pafmap_mask).get_arr()
            keypoint_mask = seq_det.augment_heatmaps(keypoint_mask).get_arr()

        pafmap_mask = cv2.resize(pafmap_mask, (self.input_shape[1] // self.downscale,
                                               self.input_shape[0] // self.downscale))
        keypoint_mask = cv2.resize(keypoint_mask, (self.input_shape[1] // self.downscale,
                                                   self.input_shape[0] // self.downscale))

        img = np.transpose(img, (2, 0, 1)).copy() / 255.
        pafmap_mask = np.transpose(pafmap_mask, (2, 0, 1)).copy()
        keypoint_mask = np.transpose(keypoint_mask, (2, 0, 1)).copy()

        return img, pafmap_mask, keypoint_mask

    def __len__(self):
        return self.dataset.shape[0]
