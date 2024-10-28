import torch
import torch.nn as nn
import torch.nn.functional as F
from net.p2t import p2t_base
import math

torch.autograd.set_detect_anomaly(True)
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IFM(nn.Module):
    def __init__(self, channel):
        super(IFM).__init__()
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forword(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c


class EEM2(nn.Module):
    def __init__(self):
        super(EEM2, self).__init__()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce4 = Conv1x1(1024, 512)
        self.encoder_block = Bottle2neck(64, 64)
        self.block = nn.Sequential(
            ConvBNR(512 + 64, 256, 3),
            ConvBNR(256, 64, 3))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x1 = self.encoder_block(x1)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)
        return out


class EEM(nn.Module):
    def __init__(self):
        super(EEM, self).__init__()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce4 = Conv1x1(512, 256)
        self.encoder_block = Bottle2neck(64, 64)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 64, 3))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x1 = self.encoder_block(x1)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)
        return out


class HIFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HIFM, self).__init__()
        # t = int(abs((log(channel, 2) + 1) / 2))
        # k = t if t % 2 else t + 1
        # self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.relu = nn.ReLU(True)
        self.fc = nn.Linear(in_channel, out_channel)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel // 4, 1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel // 4, kernel_size=3, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel // 4, kernel_size=5, padding=2)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel // 4, kernel_size=7, padding=3)
        )
        self.conv_cat = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, hf):
        if x.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, hf), dim=1)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
        b, c, _, _ = x.size()
        wei = self.avg_pool(x)
        wei = wei.squeeze()
        wei = self.fc(wei)
        wei = wei.unsqueeze(-1).unsqueeze(-1)
        wei = self.relu(wei)
        output = F.relu(wei * x_cat + self.conv_res(x))
        return output


class EIM(nn.Module):
    def __init__(self, channel):
        super(EIM, self).__init__()
        self.conv2d = ConvBNR(channel, channel, 3)
        self.reduce4 = Conv1x1(1024, 512)

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        x = self.reduce4(x)
        return x

class AMA(nn.Module):
    def __init__(self,channel):
        super(AMA, self).__init__()
        self.conv1 = ConvBNR(channel,channel)
        self.conv2 = ConvBNR(channel,channel)
    def forward(self,x):
        x = x + self.conv1(x)
        x = x + self.conv2(x)
        return x

class IMA(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(IMA, self).__init__()
        self.conv_rgb = ConvBNR(in_channel,in_channel)
        self.conv_infr = ConvBNR(in_channel,in_channel)
        self.trans = nn.Sequential(
            ConvBNR(out_channel,out_channel),
            nn.Conv2d(out_channel,2,1)
        )
    def forward(self,x_rgb,x_infr):
        x_rgb = self.conv_rgb(x_rgb)
        x_infr = self.conv_infr(x_infr)
        attn = self.trans(torch.cat([x_rgb,x_infr],dim=1))
        rgb_w, infr_w = torch.softmax(attn,dim=1).chunk(2,dim=1)
        output = torch.cat((rgb_w * x_rgb,infr_w * x_infr),dim=1)
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone1 = p2t_base(pretrained=True)
        self.backbone2 = p2t_base(pretrained=True)

        self.eem1 = EEM()
        self.eem2 = EEM2()
        self.eem3 = EEM()

        self.eim = EIM(1024)
        self.hifm3 = HIFM(1024, 512)
        self.hifm2 = HIFM(512, 256)
        self.hifm1 = HIFM(256, 128)

        self.edge_predictor = nn.Conv2d(64, 1, 1)
        self.predictor = nn.Conv2d(64, 1, 1)

        self.reduce1 = Conv1x1(128, 64)
        self.reduce2 = Conv1x1(256, 128)
        self.reduce3 = Conv1x1(640, 512)
        self.reduce4 = Conv1x1(1024, 512)

        self.conv3_1 = ConvBNR(128, 64, 3)
        self.conv3_2 = ConvBNR(256, 128, 3)
        self.conv3_3 = ConvBNR(512, 256, 3)

        self.rgb_ama1 = AMA(64)
        self.rgb_ama2 = AMA(128)
        self.rgb_ama3 = AMA(320)
        self.rgb_ama4 = AMA(512)

        self.infr_ama1 = AMA(64)
        self.infr_ama2 = AMA(128)
        self.infr_ama3 = AMA(320)
        self.infr_ama4 = AMA(512)

        self.ima1 = IMA(64,128)
        self.ima2 = IMA(128,256)
        self.ima3 = IMA(320,640)
        self.ima4 = IMA(512,1024)

        self.hidden_dim = 32
        self.input_proj_1 = nn.Sequential(nn.Conv2d(64, self.hidden_dim, kernel_size=1, bias=False),
                                             nn.BatchNorm2d(self.hidden_dim), nn.ReLU(inplace=True))
        self.input_proj_2 = nn.Sequential(nn.Conv2d(128, self.hidden_dim, kernel_size=1, bias=False),
                                             nn.BatchNorm2d(self.hidden_dim), nn.ReLU(inplace=True))
        self.input_proj_3 = nn.Sequential(nn.Conv2d(320, self.hidden_dim, kernel_size=1, bias=False),
                                             nn.BatchNorm2d(self.hidden_dim), nn.ReLU(inplace=True))
        self.input_proj_4 = nn.Sequential(nn.Conv2d(512, self.hidden_dim, kernel_size=1, bias=False),
                                               nn.BatchNorm2d(self.hidden_dim), nn.ReLU(inplace=True))

        self.rgb_mean_conv1 = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)
        self.rgb_std_conv_1  = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)

        self.rgb_mean_conv2 = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)
        self.rgb_std_conv2  = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)

        self.rgb_mean_conv3 = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)
        self.rgb_std_conv3  = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)

        self.rgb_mean_conv4 = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)
        self.rgb_std_conv4  = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)

        self.infr_mean_conv1 = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)
        self.infr_std_conv1  = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)

        self.infr_mean_conv2 = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)
        self.infr_std_conv2  = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)

        self.infr_mean_conv3 = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)
        self.infr_std_conv3  = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)

        self.infr_mean_conv4 = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)
        self.infr_std_conv4  = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)

        self.predictor1 = nn.Conv2d(64, 1, 1)

    def reparameterize(self, mu, logvar, k=1):
        sample_z = []
        for _ in range(k):
            std = logvar.mul(0.5).exp_()  # type: Variable
            eps = std.data.new(std.size()).normal_()
            sample_z.append(eps.mul(std).add_(mu))
        sample_z = torch.cat(sample_z, dim=1)
        return sample_z

    def get_weight(self,prob_out1,prob_out2,prob_out3,prob_out4):
        # print(prob_out1.shape)
        prob_out1 = (prob_out1 - prob_out1.min()) / (prob_out1.max() - prob_out1.min())
        prob_out2 = (prob_out2 - prob_out2.min()) / (prob_out2.max() - prob_out2.min())
        prob_out3 = (prob_out3 - prob_out3.min()) / (prob_out3.max() - prob_out3.min())
        prob_out4 = (prob_out4 - prob_out4.min()) / (prob_out4.max() - prob_out4.min())

        return 1 - prob_out1 , 1 - prob_out2, 1 - prob_out3, 1 - prob_out4


    def uncertainty_estimation(self,x1,x2,x3,x4):
        x1_rgb_proj = self.input_proj_1(x1)
        x1_rgb_mean = self.rgb_mean_conv1(x1_rgb_proj)
        x1_rgb_std = self.rgb_std_conv_1(x1_rgb_proj)

        x2_rgb_proj = self.input_proj_2(x2)
        x2_rgb_mean = self.rgb_mean_conv2(x2_rgb_proj)
        x2_rgb_std = self.rgb_std_conv2(x2_rgb_proj)

        x3_rgb_proj = self.input_proj_3(x3)
        x3_rgb_mean = self.rgb_mean_conv3(x3_rgb_proj)
        x3_rgb_std = self.rgb_std_conv3(x3_rgb_proj)

        x4_rgb_proj = self.input_proj_4(x4)
        x4_rgb_mean = self.rgb_mean_conv4(x4_rgb_proj)
        x4_rgb_std = self.rgb_std_conv4(x4_rgb_proj)

        prob_out_rgb1 = self.reparameterize(x1_rgb_mean, x1_rgb_std, 1)
        prob_out_rgb1 = torch.sigmoid(prob_out_rgb1)
        prob_out_rgb2 = self.reparameterize(x2_rgb_mean, x2_rgb_std, 1)
        prob_out_rgb2 = torch.sigmoid(prob_out_rgb2)
        prob_out_rgb3 = self.reparameterize(x3_rgb_mean, x3_rgb_std, 1)
        prob_out_rgb3 = torch.sigmoid(prob_out_rgb3)
        prob_out_rgb4 = self.reparameterize(x4_rgb_mean, x4_rgb_std, 1)
        prob_out_rgb4= torch.sigmoid(prob_out_rgb4)
        # print('probout')
        # print(prob_out_rgb1.shape,prob_out_rgb2.shape,prob_out_rgb3.shape,prob_out_rgb4.shape)
        wrgb1,wrgb2,wrgb3,wrgb4 = self.get_weight(prob_out_rgb1,prob_out_rgb2,prob_out_rgb3,prob_out_rgb4)
        # print("wrgb1")
        # print(wrgb1.shape,wrgb2.shape,wrgb3.shape,wrgb4.shape)
        return prob_out_rgb1,prob_out_rgb2,prob_out_rgb3,prob_out_rgb4,wrgb1,wrgb2,wrgb3,wrgb4
    
    def get_norm_weight(self,wrgb,winfr):
        return (wrgb + 1e-8) /(wrgb + winfr + 1e-8), (winfr + 1e-8)/(wrgb + winfr+1e-8)

    def forward(self, x_rgb, x_infr):
        x_rgb_size = x_rgb.size()
        x_rgb_h,x_rgb_w = x_rgb_size[2],x_rgb_size[3]
        x1_rgb, x2_rgb, x3_rgb, x4_rgb = self.backbone1(x_rgb)
        x1_infr, x2_infr, x3_infr, x4_infr = self.backbone2(x_infr)

        prob_out_rgb1,prob_out_rgb2,prob_out_rgb3,prob_out_rgb4, wrgb1, wrgb2, wrgb3, wrgb4, = self.uncertainty_estimation(x1_rgb,x2_rgb,x3_rgb,x4_rgb)
        prob_out_infr1,prob_out_infr2,prob_out_infr3,prob_out_infr4,winfr1,winfr2,winfr3,winfr4 = self.uncertainty_estimation(x1_infr,x2_infr,x3_infr,x4_infr)

        wrgb1,winfr1 = self.get_norm_weight(wrgb1,winfr1)
        wrgb2,winfr2 = self.get_norm_weight(wrgb2,winfr2)
        wrgb3,winfr3 = self.get_norm_weight(wrgb3,winfr3)
        wrgb4,winfr4 = self.get_norm_weight(wrgb4,winfr4)
        # print(x1_rgb.shape,wrgb1.shape)
        x1_rgb_fuse = self.rgb_ama1(x1_rgb * wrgb1)
        x2_rgb_fuse = self.rgb_ama2(x2_rgb * wrgb2)
        x3_rgb_fuse = self.rgb_ama3(x3_rgb * wrgb3)
        x4_rgb_fuse = self.rgb_ama4(x4_rgb * wrgb4)

        x1_infr_fuse = self.infr_ama1(x1_infr * winfr1)
        x2_infr_fuse = self.infr_ama2(x2_infr * winfr2)
        x3_infr_fuse = self.infr_ama3(x3_infr * winfr3)
        x4_infr_fuse = self.infr_ama4(x4_infr * winfr4)

        x1_fuse = self.ima1(x1_rgb_fuse,x1_infr_fuse)
        x2_fuse = self.ima2(x2_rgb_fuse,x2_infr_fuse)
        x3_fuse = self.ima3(x3_rgb_fuse,x3_infr_fuse)
        x4_fuse = self.ima4(x4_rgb_fuse,x4_infr_fuse)

        # x1_fuse = torch.cat((wrgb1 * x1_rgb, winfr1 * x1_infr), dim=1)  # TODO 128
        # x2_fuse = torch.cat((wrgb2 * x2_rgb, winfr2 * x2_infr), dim=1)  # 256
        # x3_fuse = torch.cat((wrgb3 * x3_rgb, winfr3 * x3_infr), dim=1)  # 640
        # x4_fuse = torch.cat((wrgb4 * x4_rgb, winfr4 * x4_infr), dim=1)  # 1024


        edge_rgb = self.eem1(x4_rgb, x1_rgb)
        edge_fuse = self.eem2(x4_fuse, edge_rgb)
        edge_infr = self.eem3(x4_infr, edge_fuse)
        edge_infr = self.edge_predictor(edge_infr)

        edge_att = torch.sigmoid(edge_infr)

        x4a = self.eim(x4_fuse, edge_att)  # 512

        x3_fuse = self.reduce3(x3_fuse)  # 512

        x3r = self.hifm3(x3_fuse, x4a)  # 512 512
        x3r = self.conv3_3(x3r)  # 256
        x2r = self.hifm2(x2_fuse, x3r)  # 256 256 -> 256
        x2r = self.conv3_2(x2r)  # 128
        x1r = self.hifm1(x1_fuse, x2r)  # 128 128 -> 128
        x1r = self.conv3_1(x1r)  # 128->64
        o1 = self.predictor(x1r)  # 1
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        prob_out_rgb1 = F.interpolate(prob_out_rgb1, size=(x_rgb_h, x_rgb_w), mode='bilinear', align_corners=True)
        prob_out_rgb2 = F.interpolate(prob_out_rgb2, size=(x_rgb_h, x_rgb_w), mode='bilinear', align_corners=True)
        prob_out_rgb3 = F.interpolate(prob_out_rgb3, size=(x_rgb_h, x_rgb_w), mode='bilinear', align_corners=True)
        prob_out_rgb4 = F.interpolate(prob_out_rgb4, size=(x_rgb_h, x_rgb_w), mode='bilinear', align_corners=True)
        prob_out_infr1 = F.interpolate(prob_out_infr1, size=(x_rgb_h, x_rgb_w), mode='bilinear', align_corners=True)
        prob_out_infr2 = F.interpolate(prob_out_infr2, size=(x_rgb_h, x_rgb_w), mode='bilinear', align_corners=True)
        prob_out_infr3 = F.interpolate(prob_out_infr3, size=(x_rgb_h, x_rgb_w), mode='bilinear', align_corners=True)
        prob_out_infr4 = F.interpolate(prob_out_infr4, size=(x_rgb_h, x_rgb_w), mode='bilinear', align_corners=True)

        return o1, oe, prob_out_rgb1,prob_out_rgb2,prob_out_rgb3,prob_out_rgb4,prob_out_infr1,prob_out_infr2,prob_out_infr3,prob_out_infr4
