import torch.nn as nn
import torch.nn.functional as F


class VGGHead(nn.Module):
    def __init__(self):
        super(VGGHead, self).__init__()

        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv_0_0 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv_0_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv_0_2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_0_3 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv_1_0 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_1_1 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv_2_0 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_2_1 = nn.Conv2d(128, 128, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv_0_0(x))
        x = F.relu(self.conv_0_1(x))
        x = F.relu(self.conv_0_2(x))
        x = self.maxpool(F.relu(self.conv_0_3(x)))

        x = F.relu(self.conv_1_0(x))
        x = self.maxpool(F.relu(self.conv_1_1(x)))

        x = F.relu(self.conv_2_0(x))
        x = self.maxpool(F.relu(self.conv_2_1(x)))

        return x


class ResBlock(nn.Module):
    def __init__(self, inc, outc):
        super(ResBlock, self).__init__()

        self.conv_skip = nn.Conv2d(inc, outc, 1)

        self.conv_res_0 = nn.Conv2d(inc, outc // 2, 1)
        self.conv_res_1 = nn.Conv2d(inc, outc // 2, 3, padding=1)
        self.conv_res_2 = nn.Conv2d(inc, outc, 1)

    def forward(self, x):
        skip = F.relu(self.conv_skip(x))

        res = F.relu(self.conv_res_0(x))
        res = F.relu(self.conv_res_1(res))
        res = F.relu(self.conv_res_2(res))

        out = skip + res

        return out


class ResBlockX3(nn.Module):
    def __init__(self, inc, outc):
        super(ResBlockX3, self).__init__()

        self.res_block_0 = ResBlock(inc, outc)
        self.res_block_1 = ResBlock(inc, outc)
        self.res_block_2 = ResBlock(inc, outc)

    def forward(self, x):
        x = self.res_block_0.forward(x)
        x = self.res_block_1.forward(x)
        x = self.res_block_2.forward(x)

        return x


class HourglassBlock(nn.Module):
    def __init__(self, inc, depth=3):
        super(HourglassBlock, self).__init__()

        self.depth = depth
        self.maxpool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.blocks_down = [ResBlockX3(inc * 2 ** d, inc * 2 ** d) for d in range(depth)]
        self.skips = [ResBlockX3(inc * 2 ** d, inc * 2 ** d) for d in range(depth)]
        self.blocks_up = [ResBlockX3(inc * 2 ** (depth - d - 1), inc * 2 ** (depth - d - 1)) for d in range(depth)]

        self.bottleneck = ResBlockX3(inc * 2 ** depth, inc * 2 ** depth)

    def forward(self, x):
        current_in = x
        skips = []
        for i in range(self.depth):
            current_in = self.blocks_down[i].forward(current_in)
            skips.append(self.skips[i].forward(current_in))
            current_in = self.maxpool(current_in)

        bottleneck = self.bottleneck.forward(current_in)

        current_in = bottleneck
        for i in range(self.depth):
            current_in = self.upsample(current_in)
            current_in = skips[self.depth - i - 1] + current_in
            current_in = self.blocks_up[i].forward(current_in)

        return current_in
