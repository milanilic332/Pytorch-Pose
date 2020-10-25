import configparser
import torch
import numpy as np
from tqdm import tqdm
from ast import literal_eval
import cv2
import scipy.ndimage as sp

from src.networks.network_factory import build_network
from src.dataset.dataloader_factory import build_dataloaders


class HourglassVisualizer:
    def __init__(self, config):
        self.config = config

        self.input_shape = literal_eval(self.config['default']['input_shape'])
        self.kps = literal_eval(self.config['network']['coco_keypoint_points'])
        self.pafs = literal_eval(self.config['network']['coco_paf_joints'])
        self.hourglass_stages = self.config['network'].getint(['hourglass_stages'])
        self.joints_id = []
        for i, (paf0, paf1) in enumerate(self.pafs):
            self.joints_id.append([self.kps.index(paf0), self.kps.index(paf1)])

        self.downscale = self.config['default'].getint('downscale')

        self.model = build_network(config)
        self.model.load_state_dict(torch.load('../saved/models/HourglassV1_best.pth'))

        self.video_cap = cv2.VideoCapture('../data/test_videos/me.MOV')

    def visualize(self):
        with torch.no_grad():
            while True:
                ret, frame = self.video_cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))
                frame = np.transpose(frame, (2, 0, 1)) / 255.
                frame = np.expand_dims(frame, 0)
                frame = torch.from_numpy(frame).float().cuda()

                o1, o2 = self.model(frame)

                frame, o1, o2 = (np.transpose(frame.cpu().numpy(), (0, 2, 3, 1))[0],
                                 np.transpose(o1[self.hourglass_stages - 1].cpu().numpy(), (0, 2, 3, 1)),
                                 np.transpose(o2[self.hourglass_stages - 1].cpu().numpy(), (0, 2, 3, 1)))

                new_o1 = np.zeros((*o1.shape[:3], o1.shape[3] // 2))
                for i in range(o1.shape[3] // 2):
                    new_o1[..., i] = np.clip(np.abs(o1[..., i * 2] + o1[..., i * 2 + 1]), 0, 1)

                cv2.imshow('img', frame)
                cv2.imshow('paf_pred', cv2.resize(np.max(new_o1[0], axis=2), (frame.shape[1], frame.shape[0])))
                cv2.imshow('kp_pred', cv2.resize(np.max(o2[0], axis=2), (frame.shape[1], frame.shape[0])))
                cv2.waitKey(1)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('./train.cfg')

    visualizer = HourglassVisualizer(config)
    visualizer.visualize()
