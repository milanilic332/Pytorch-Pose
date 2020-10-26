import torch
import torch.nn as nn

from src.networks.common_blocks import VGGHead, ResBlockX3


class ResNetKP(nn.Module):
    def __init__(self, n_pafs, n_keypoints):
        """Resnet Pose network

        :param n_pafs:              number of paf outputs
        :param n_keypoints:         number of keypoint outputs
        """
        super(ResNetKP, self).__init__()

        self.vgg = VGGHead()

        self.paf_0 = ResBlockX3(64, 64)
        self.paf_1 = ResBlockX3(64, 64)
        self.paf_2 = ResBlockX3(64, 64)

        self.paf_prep = ResBlockX3(64, 32)
        self.paf_out = nn.Conv2d(32, n_pafs, 3, padding=1)

        self.class_0 = ResBlockX3(128, 64)
        self.class_1 = ResBlockX3(64, 64)
        self.class_2 = ResBlockX3(64, 64)

        self.class_prep = ResBlockX3(64, 32)
        self.class_out = nn.Conv2d(32, n_keypoints, 3, padding=1)

    def forward(self, x):
        vgg = self.vgg.forward(x)

        hp = self.paf_0.forward(vgg)
        hp = self.paf_1.forward(hp)
        hp = self.paf_2.forward(hp)

        hpo = self.paf_prep.forward(hp)
        hpo = torch.sigmoid(self.paf_out(hpo))

        class_in = torch.cat((vgg, hp), 1)

        hc = self.class_0.forward(class_in)
        hc = self.class_1.forward(hc)
        hc = self.class_2.forward(hc)

        hco = self.class_prep.forward(hc)
        hco = torch.sigmoid(self.class_out(hco))

        return hpo, hco
