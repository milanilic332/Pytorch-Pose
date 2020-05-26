import torch
import torch.nn as nn

from src.networks.common_blocks import VGGHead, ResBlockX3, HourglassModule


class HourglassV1(nn.Module):
    def __init__(self, paf_joints, keypoint_points, stages=3):
        super(HourglassV1, self).__init__()

        self.stages = stages

        self.vgg = VGGHead().cuda()

        self.pafs = nn.ModuleList([HourglassModule(64, 64, 3).cuda() for _ in range(self.stages)])
        self.paf_prep_outs = nn.ModuleList([ResBlockX3(64, 32).cuda() for _ in range(self.stages)])
        self.paf_outs = nn.ModuleList([nn.Conv2d(32, len(paf_joints), 3, padding=1).cuda() for _ in range(self.stages)])

        self.class_prep = nn.Conv2d(128, 64, 1, padding=0).cuda()

        self.classes = nn.ModuleList([HourglassModule(64, 64, 3).cuda() for _ in range(self.stages)])
        self.class_prep_outs = nn.ModuleList([ResBlockX3(64, 32).cuda() for _ in range(self.stages)])
        self.class_outs = nn.ModuleList([nn.Conv2d(32, len(keypoint_points), 3, padding=1).cuda() for _ in range(self.stages)])

    def forward(self, x):
        vgg = self.vgg.forward(x)
        x = vgg

        pos = []
        for s in range(self.stages):
            x = self.pafs[s].forward(x)
            pos.append(torch.sigmoid(self.paf_outs[s](self.paf_prep_outs[s].forward(x))))

            if s != self.stages - 1:
                x = vgg + x

        class_input = torch.relu(self.class_prep(torch.cat((vgg, x), 1)))
        x = class_input

        cos = []
        for s in range(self.stages):
            x = self.classes[s].forward(x)
            cos.append(torch.sigmoid(self.class_outs[s](self.class_prep_outs[s].forward(x))))

            if s != self.stages - 1:
                x = class_input + x

        return pos, cos
