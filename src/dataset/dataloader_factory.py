from ast import literal_eval
from time import time
from torch.utils.data import DataLoader

from src.dataset.datasets import PoseDataset


def build_dataloaders(config):
    train_dataset = PoseDataset(type='train',
                                dataset_root=config['dataset']['root'],
                                pafmap_joints=[literal_eval(config['network']['coco_paf_joints']),
                                               literal_eval(config['network']['mpii_paf_joints'])],
                                keypoints=[literal_eval(config['network']['coco_keypoint_points']),
                                           literal_eval(config['network']['mpii_keypoint_points'])],
                                augment=config['default'].getboolean('augmentation'),
                                downscale=config['default'].getint('downscale'))


    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config['default'].getint('batch_size'),
                                  shuffle=True,
                                  num_workers=16,
                                  pin_memory=True)

    if config['default'].getboolean('validation'):
        val_dataset = PoseDataset(type='val',
                                  dataset_root=config['dataset']['root'],
                                  pafmap_joints=[literal_eval(config['network']['coco_paf_joints']),
                                                 literal_eval(config['network']['mpii_paf_joints'])],
                                  keypoints=[literal_eval(config['network']['coco_keypoint_points']),
                                             literal_eval(config['network']['mpii_keypoint_points'])],
                                  augment=False,
                                  downscale=config['default'].getint('downscale'))

        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=config['default'].getint('batch_size'),
                                    shuffle=False,
                                    num_workers=16,
                                    pin_memory=True)

    return [train_dataloader, val_dataloader if config['default'].get('validation') else None]
