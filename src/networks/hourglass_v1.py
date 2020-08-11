import torch
import torch.nn as nn

from src.networks.common_blocks import VGGHead, HourglassModule, ResBlock


class HourglassV1(nn.Module):
    def __init__(self, n_pafs, n_keypoints, stages=5):
        super(HourglassV1, self).__init__()

        self.stages = stages

        self.vgg = VGGHead().cuda()

        self.paf_preps = nn.ModuleList([ResBlock(128, 64).cuda() for _ in range(self.stages)])
        self.pafs = nn.ModuleList([HourglassModule(64, 64, 3).cuda() for _ in range(self.stages)])
        self.paf_outs = nn.ModuleList([nn.Conv2d(64, n_pafs * 2, 3, padding=1).cuda() for _ in range(self.stages)])

        self.class_ins = ResBlock(128, 64).cuda()

        self.class_preps = nn.ModuleList([ResBlock(128, 64).cuda() for _ in range(self.stages)])
        self.classes = nn.ModuleList([HourglassModule(64, 64, 3).cuda() for _ in range(self.stages)])
        self.class_outs = nn.ModuleList([nn.Conv2d(64, n_keypoints, 3, padding=1).cuda() for _ in range(self.stages)])

    def forward(self, x):
        vgg = self.vgg.forward(x)
        x = vgg

        pos = []
        for s in range(self.stages):
            x = self.pafs[s].forward(x)
            pos.append(self.paf_outs[s](x))

            if s != self.stages - 1:
                x = torch.cat((vgg, x), 1)
                x = torch.relu(self.paf_preps[s].forward(x))

        class_input = torch.relu(self.class_ins.forward(torch.cat((vgg, x), 1)))
        x = class_input

        cos = []
        for s in range(self.stages):
            x = self.classes[s].forward(x)
            cos.append(torch.sigmoid(self.class_outs[s](x)))

            if s != self.stages - 1:
                x = torch.cat((class_input, x), 1)
                x = torch.relu(self.class_preps[s].forward(x))

        return pos, cos
