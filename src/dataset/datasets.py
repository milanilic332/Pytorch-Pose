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
    def __init__(self, type, dataset_root, pafmap_joints, keypoints,
                 input_shape=(288, 512, 3), augment=False, downscale=4):
        """Pytorch dataset class handling coco and mpii data

        :param type:                'train' or 'validation'
        :param dataset_root:        root path of datasets
        :param pafmap_joints:       mapping list of joints
        :param keypoints:           mapping list of keypoints
        :param input_shape:         input shape of images
        :param augment:             to augment or not
        :param downscale:           downscale factor of output
        """
        self.type = type
        self.dataset_root = dataset_root
        self.input_shape = input_shape
        self.augment = augment

        if self.augment:
            self.seq = get_augmentations()

        self.coco_pafmap_joints = pafmap_joints[0]
        self.coco_keypoints = keypoints[0]
        self.mpii_pafmap_joints = pafmap_joints[1]
        self.mpii_keypoints = keypoints[1]

        self.downscale = downscale
        
        self.keypoint_mask = get_keypoint_mask()

        self.dataset = self.load_coco_ann_files()

    def load_coco_ann_files(self):
        """Loading COCO and MPII data

        :return:            dataset as pd.DataFrame object
        """
        if self.type == 'train':
            datasets = [
                        (os.path.join(self.dataset_root, 'coco', 'train2014'),
                         COCO(os.path.join(self.dataset_root, 'coco',
                                           'annotations_trainval2014', 'person_keypoints_train2014.json'))),
                        (os.path.join(self.dataset_root, 'coco', 'train2017'),
                         COCO(os.path.join(self.dataset_root, 'coco',
                                           'annotations_trainval2017', 'person_keypoints_train2017.json'))),
                        # (os.path.join(self.dataset_root, 'mpii', 'images'),
                        #  COCO(os.path.join(self.dataset_root, 'mpii',
                        #                    'annotations', 'train.json')))
                       ]
        else:
            datasets = [
                        (os.path.join(self.dataset_root, 'coco', 'val2014'),
                         COCO(os.path.join(self.dataset_root, 'coco',
                                           'annotations_trainval2014', 'person_keypoints_val2014.json'))),
                        (os.path.join(self.dataset_root, 'coco', 'val2017'),
                         COCO(os.path.join(self.dataset_root, 'coco',
                                           'annotations_trainval2017', 'person_keypoints_val2017.json')))
                       ]

        dict_list = []
        for dataset_path, dataset in datasets:
            img_ids = dataset.getImgIds()

            for idx in img_ids:
                try:
                    img = dataset.loadImgs([idx])[0]
                    ann_ids = dataset.getAnnIds([idx])
                    anns = dataset.loadAnns(ann_ids)

                    if [ann['keypoints'] for ann in anns] and not all([ann['keypoints'] == [0]*51 for ann in anns]):
                        keypoints = [ann['keypoints'] for ann in anns if ann['keypoints'] != [0]*51]
                        for i in range(len(keypoints)):
                            if 'coco' in dataset_path:
                                keypoints[i] = keypoints[i] + ([0, 0, 0] if not (keypoints[i][17] and keypoints[i][20])
                                               else [(keypoints[i][15] + keypoints[i][18]) // 2, (keypoints[i][16] + keypoints[i][19]) // 2, 1])
                            else:
                                keypoints[i] = keypoints[i] + ([0, 0, 0] if not (keypoints[i][41] and keypoints[i][38])
                                               else [(keypoints[i][39] + keypoints[i][36]) // 2, (keypoints[i][40] + keypoints[i][37]) // 2, 1])

                        if len([kp for kp in keypoints if kp != [0]*54]) <= 4:
                            dict_list.append({'path': os.path.join(dataset_path, img["file_name"]),
                                              'keypoints': [kp for kp in keypoints if kp != [0]*54]})
                except:
                    print(f'Skipped: {idx}')

        final_dataset = pd.DataFrame.from_dict(dict_list)

        return final_dataset

    def apply_keypoint_mask(self, img_keypoints, input_shape, keypoints, kp_size=32):
        """Create keypoint gt masks

        :param img_keypoints:           keypoints from pandas dataset
        :param input_shape:             input shape of images
        :param keypoints:               keypoints mapping for coco or mpii
        :param kp_size:                 size of kp 2d mask
        :return:                        keypoint outputs, number of keypoints for each keypoint
        """
        keypoint_masks = [np.zeros((self.input_shape[0] + kp_size, self.input_shape[1] + kp_size, 1), dtype=np.float32)
                          for _ in range(len(keypoints))]
        n_keypoints = [0 for _ in range(len(keypoints))]

        for kp in img_keypoints:
            for i, k in enumerate(keypoints):
                x = int(np.round(kp[k * 3] / input_shape[1] * self.input_shape[1]))
                y = int(np.round(kp[k * 3 + 1] / input_shape[0] * self.input_shape[0]))
                if kp[k * 3 + 2] != 0:
                    try:
                        keypoint_masks[i][y:y + kp_size, x:x + kp_size, 0] = \
                            np.maximum(keypoint_masks[i][y:y + kp_size, x:x + kp_size, 0], self.keypoint_mask)
                        n_keypoints[i] += 1
                    except:
                        pass

        keypoint_masks = keypoint_masks if np.max(keypoint_masks) < 0.5 else keypoint_masks / np.max(keypoint_masks)

        return np.squeeze(np.array(keypoint_masks)).transpose((1, 2, 0))[kp_size // 2:-kp_size // 2, kp_size // 2:-kp_size // 2, :], np.array(n_keypoints)

    def apply_pafmap_mask(self, img_keypoints, input_shape, pafmap_joints, thickness=8):
        """Create pafs gt masks

        :param img_keypoints:           keypoints from pandas dataset
        :param input_shape:             input shape of images
        :param pafmap_joints:           paf mapping for coco or mpii
        :param thickness:               thickness of joint lines
        :return:                        pafmap outputs, number of pafs for each joint
        """
        paf_masks = [np.zeros((self.input_shape[0], self.input_shape[1], 1), dtype=np.float32)
                     for _ in range(len(pafmap_joints) * 2)]
        n_pafs = [0 for _ in range(len(pafmap_joints))]

        for kp in img_keypoints:
            for i, (kp0, kp1) in enumerate(pafmap_joints):
                x1 = int(np.round(kp[kp0 * 3] / input_shape[1] * self.input_shape[1]))
                y1 = int(np.round(kp[kp0 * 3 + 1] / input_shape[0] * self.input_shape[0]))
                x2 = int(np.round(kp[kp1 * 3] / input_shape[1] * self.input_shape[1]))
                y2 = int(np.round(kp[kp1 * 3 + 1] / input_shape[0] * self.input_shape[0]))

                if kp[kp0 * 3 + 2] != 0 and kp[kp1 * 3 + 2] != 0 and (x2 - x1 != 0 or y2 - y1 != 0):
                    tmp_paf_x = np.zeros((self.input_shape[0], self.input_shape[1], 1), dtype=np.float32)
                    tmp_paf_y = np.zeros((self.input_shape[0], self.input_shape[1], 1), dtype=np.float32)

                    cv2.line(tmp_paf_x, (x1, y1), (x2, y2), 1, thickness)
                    cv2.line(tmp_paf_y, (x1, y1), (x2, y2), 1, thickness)

                    x_val = (x2 - x1) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    y_val = (y2 - y1) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                    tmp_paf_x = tmp_paf_x * x_val
                    tmp_paf_y = tmp_paf_y * y_val

                    paf_masks[i * 2][tmp_paf_x != 0] = tmp_paf_x[tmp_paf_x != 0]
                    paf_masks[i * 2 + 1][tmp_paf_y != 0] = tmp_paf_y[tmp_paf_y != 0]

                    n_pafs[i] += 1

        paf_masks = np.array(paf_masks)

        return np.squeeze(paf_masks).transpose((1, 2, 0)), np.array(n_pafs)

    def __getitem__(self, idx):
        """Gets the next data

        :param idx:             id of element inside the pandas dataset
        :return:                input image, pafmap gt, keypoint gt, number of pafs, number of keypoints
        """
        row = self.dataset.iloc[idx]

        img = cv2.imread(row['path'])
        img_keypoints = row['keypoints']

        pafmap_mask, n_pafs = self.apply_pafmap_mask(img_keypoints, img.shape,
                                                     self.coco_pafmap_joints if 'coco' in row['path'] else self.mpii_pafmap_joints)

        keypoint_mask, n_kps = self.apply_keypoint_mask(img_keypoints, img.shape,
                                                        self.coco_keypoints if 'coco' in row['path'] else self.mpii_keypoints)

        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))

        if self.augment:
            seq_det = self.seq.to_deterministic()

            pafmap_mask = HeatmapsOnImage(pafmap_mask, shape=img.shape, min_value=-1.0, max_value=1.0)

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

        return img, pafmap_mask, keypoint_mask, n_pafs, n_kps

    def __len__(self):
        """Length of dataset getter

        :return:        dataset length
        """
        return self.dataset.shape[0]
