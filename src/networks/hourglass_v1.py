import torch
import torch.nn as nn
import torch.nn.functional as F

from src.networks.common_blocks import VGGHead, ResBlockX3, HourglassBlock


class HourglassV1(nn.Module):
    def __init__(self, hourglass_channels, paf_outputs=10, class_outputs=9):
        super(HourglassV1, self).__init__()

        self.vgg = VGGHead()

        self.hourglass_paf_prep = nn.Conv2d(128, hourglass_channels, 1)

        self.hourglass_paf_0 = HourglassBlock(32)
        self.hourglass_paf_1 = HourglassBlock(32)
        self.hourglass_paf_2 = HourglassBlock(32)

        self.paf_prep = ResBlockX3(32, 32)
        self.paf_out = nn.Conv2d(32, paf_outputs, 3, padding=1)

        self.hourglass_class_prep = nn.Conv2d(128, hourglass_channels, 1)

        self.hourglass_class_0 = HourglassBlock(32)
        self.hourglass_class_1 = HourglassBlock(32)
        self.hourglass_class_2 = HourglassBlock(32)

        self.class_prep = ResBlockX3(32, 32)
        self.class_out = nn.Conv2d(32, class_outputs, 3, padding=1)

    def forward(self, x):
        vgg = self.vgg.forward(x)

        hp = F.relu(self.hourglass_paf_prep(x))
        hp = self.hourglass_paf_0.forward(hp)
        hp = self.hourglass_paf_1.forward(hp)
        hp = self.hourglass_paf_2.forward(hp)
        hpo = self.paf_prep.forward(hp)
        hpo = F.relu(self.paf_out(hpo))

        class_in = torch.cat((vgg, hp), 0)

        hc = F.relu(self.hourglass_class_prep(class_in))
        hc = self.hourglass_class_0.forward(hc)
        hc = self.hourglass_class_1.forward(hc)
        hc = self.hourglass_class_2.forward(hc)
        hco = self.class_prep.forward(hc)
        hco = F.relu(self.class_out(hco))

        return hpo, hco
