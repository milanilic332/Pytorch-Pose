import torch.nn as nn
import torch.nn.functional as F


class VGGHead(nn.Module):
    def __init__(self):
        super(VGGHead, self).__init__()

        self.maxpool = nn.MaxPool2d(2, 2).cuda()

        self.rb_0_0 = ResBlock(3, 32).cuda()
        self.rb_0_1 = ResBlock(32, 32).cuda()

        self.rb_1_0 = ResBlock(32, 48).cuda()
        self.rb_1_1 = ResBlock(48, 48).cuda()

        self.rb_2_0 = ResBlock(48, 64).cuda()
        self.rb_2_1 = ResBlock(64, 64).cuda()

    def forward(self, x):
        x = self.rb_0_0.forward(x)
        x = self.maxpool(self.rb_0_1.forward(x))

        x = self.rb_1_0.forward(x)
        x = self.maxpool(self.rb_1_1.forward(x))

        x = self.rb_2_0.forward(x)
        x = self.rb_2_1.forward(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, inc, outc):
        super(ResBlock, self).__init__()

        self.conv_skip = nn.Conv2d(inc, outc, 1).cuda()

        self.conv_res_0 = nn.Conv2d(inc, outc // 2, 1).cuda()
        self.conv_res_1 = nn.Conv2d(outc // 2, outc // 2, 3, padding=1).cuda()
        self.conv_res_2 = nn.Conv2d(outc // 2, outc, 1).cuda()

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

        self.res_block_0 = ResBlock(inc, outc).cuda()
        self.res_block_1 = ResBlock(outc, outc).cuda()
        self.res_block_2 = ResBlock(outc, outc).cuda()

    def forward(self, x):
        x = self.res_block_0.forward(x)
        x = self.res_block_1.forward(x)
        x = self.res_block_2.forward(x)

        return x


class HourglassModule(nn.Module):
    def __init__(self, inc, outc, pools):
        super(HourglassModule, self).__init__()

        self.pools = pools

        self.preprocess = ResBlock(inc, outc).cuda()

        self.downs = nn.ModuleList([ResBlockX3(inc, outc).cuda() for _ in range(self.pools)])
        self.maxpool = nn.MaxPool2d(2).cuda()
        self.skips = nn.ModuleList([ResBlockX3(inc, outc).cuda() for _ in range(self.pools)])
        self.upsample = nn.Upsample(scale_factor=2).cuda()
        self.ups = nn.ModuleList([ResBlockX3(inc, outc).cuda() for _ in range(self.pools)])
        self.bottleneck = ResBlockX3(inc, outc).cuda()

        self.postprocess = ResBlock(inc, outc).cuda()

    def forward(self, x):
        x = self.preprocess.forward(x)

        skips = []
        for d in range(self.pools):
            x = self.maxpool(x)
            skips.append(self.skips[d].forward(x))
            x = self.downs[d].forward(x)

        x = self.bottleneck.forward(x)

        for d in range(self.pools):
            x = x + skips[self.pools - d - 1]
            x = self.ups[d].forward(x)
            x = self.upsample(x)

        x = self.postprocess.forward(x)

        return x


