import configparser
import torch
import numpy as np
from tqdm import tqdm
import cv2

from src.networks.network_factory import build_network
from src.dataset.dataloader_factory import build_dataloaders


class Visualizer:
    def __init__(self, config):
        self.config = config

        self.model = build_network(config)
        self.model.load_state_dict(torch.load('../saved/models/HourglassV1_0.pth'))

        self.train_dataloader, self.val_dataloader = build_dataloaders(config)

    def visualize(self):
        with torch.no_grad():
            for data in tqdm(self.val_dataloader):
                imgs, pafs, kps = data
                imgs, pafs, kps = imgs.cuda(), pafs.cuda(), kps.cuda()

                o1, o2 = self.model(imgs.float())

                imgs, pafs, kps, o1, o2 = (np.transpose(imgs.cpu().numpy(), (0, 2, 3, 1)),
                                           np.transpose(pafs.cpu().numpy(), (0, 2, 3, 1)),
                                           np.transpose(kps.cpu().numpy(), (0, 2, 3, 1)),
                                           np.transpose(o1[2].cpu().numpy(), (0, 2, 3, 1)),
                                           np.transpose(o2[2].cpu().numpy(), (0, 2, 3, 1)))


                cv2.imshow('img', imgs[0])
                cv2.imshow('paf_true', cv2.resize(np.max(pafs[0], axis=2), (imgs[0].shape[1], imgs[0].shape[0])))
                cv2.imshow('kp_true', cv2.resize(np.max(kps[0], axis=2), (imgs[0].shape[1], imgs[0].shape[0])))
                cv2.imshow('paf_pred', cv2.resize(np.max(o1[0], axis=2), (imgs[0].shape[1], imgs[0].shape[0])))
                cv2.imshow('kp_pred', cv2.resize(np.max(o2[0], axis=2), (imgs[0].shape[1], imgs[0].shape[0])))
                cv2.waitKey(0)

                break


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('./train.cfg')

    visualizer = Visualizer(config)
    visualizer.visualize()
