import torch
from torch import nn
from torchvision import models
from utils import unfold
import torch.nn.functional as F


def get_parameter_num(model):
    total_num = sum(x.numel() for x in model.parameters())
    print("Total parameter number of the model is: %.2fM" % (total_num / 1e6))
    return total_num


class low_rank_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fea_tea, fea_stu):
        n = fea_stu.size(0)
        loss = 0
        for i in range(n):
            tea = fea_tea[i, :, :, :]
            stu = fea_stu[i, :, :, :]
            con = torch.cat([tea, stu], dim=0)
            uf_con = unfold(con, 1)
            loss = loss + torch.norm(uf_con, p='nuc') / n
        return loss


class VGGNet(nn.Module):
    def __init__(self, bands=3, num_class=2, mode='NKD', ret='single',
                 srrl=False):  # NKD -- not in knowledge distillation mode KD -- in knowledge distillation mode
        # single -- return features output by last conv  all -- return all features
        super(VGGNet, self).__init__()
        self.mode = mode
        self.ret = ret
        self.srrl = srrl
        net = models.vgg16(pretrained=False)
        net.classifier = nn.Sequential()
        net.features[0] = nn.Conv2d(bands, 64, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(size=(32, 32))  # the minimum input shape of vgg16 is 32*32
        self.encoder = net
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_class),
        )

    def forward(self, x):
        x_temp = x
        x = self.upsample(x)
        x = self.encoder(x)
        if self.mode == 'KD':
            y = x.reshape([-1, 512, 7, 7])
        x = x.view(x.size(0), -1)
        w = x
        x = self.classifier(x)
        if self.mode == 'KD':
            if self.srrl == True:
                return x, w
            if self.ret == 'single':
                return x, y
            else:
                z = []
                x_temp = self.upsample(x_temp)
                for name, module in self.encoder.features.named_children():
                    x_temp = module(x_temp)
                    if name == '16':
                        z.append(x_temp)
                        break
                z.append(y)
                return x, z
        else:
            return x


class StudentVGGNet(nn.Module):  # improved VGG-8, fewer parameter for higher compressed ratio
    def __init__(self, bands=3, num_class=2, mode='NKD', ret='single', srrl=False):
        super(StudentVGGNet, self).__init__()
        self.ret = ret
        self.srrl = srrl
        self.mode = mode
        self.upsample = nn.Upsample(size=(32, 32))
        self.encoder = nn.Sequential(
            nn.Conv2d(bands, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_class),
        )
        # for name, module in self.encoder.named_children():
        #     print(name)

    def forward(self, x):
        x_temp = x
        x = self.upsample(x)
        x = self.encoder(x)
        if self.mode == 'KD':
            y = x
        x = x.view(x.size(0), -1)
        w = x
        x = self.classifier(x)
        if self.mode == 'KD':
            if self.srrl == True:
                return x, w
            if self.ret == 'single':
                return x, y
            else:
                z = []
                x_temp = self.upsample(x_temp)
                for name, module in self.encoder.named_children():
                    x_temp = module(x_temp)
                    if name == '8':
                        z.append(x_temp)
                        break
                z.append(y)
                return x, z
        else:
            return x


class ResNet50(nn.Module):
    def __init__(self, bands=3, num_class=2, mode='NKD', ret='single', srrl=False):
        super(ResNet50, self).__init__()
        self.mode = mode
        self.ret = ret
        self.srrl = srrl
        net = models.resnet50(pretrained=False)
        net.conv1 = nn.Conv2d(bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.avgpool = nn.Sequential()
        net.fc = nn.Sequential()
        self.net = net
        self.upsample = nn.Upsample(size=(32, 32))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))
        self.fc = nn.Sequential(
            nn.Linear(2048 * 3 * 3, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_class),
        )

    def forward(self, x):
        x_temp = x
        x = self.upsample(x)
        x = self.net(x)
        x = x.reshape([x.shape[0], x.shape[1], 1, 1])
        x = self.avgpool(x)
        if self.mode == 'KD':
            y = x.reshape([-1, 2048, 3, 3])
        x = x.view(x.size(0), -1)
        w = x
        x = self.fc(x)
        if self.mode == 'KD':
            if self.srrl == True:
                return x, w
            if self.ret == 'single':
                return x, y
            else:
                z = []
                x_temp = self.upsample(x_temp)
                for name, module in self.net.named_children():
                    x_temp = module(x_temp)
                    if name == 'layer2':
                        z.append(x_temp)
                        break
                z.append(y)
                return x, z
        else:
            return x


class ResNet18(nn.Module):
    def __init__(self, bands=3, num_class=2, mode='NKD', ret='single', srrl=False):
        super(ResNet18, self).__init__()
        self.mode = mode
        self.ret = ret
        self.srrl = srrl
        net = models.resnet18(pretrained=False)
        net.conv1 = nn.Conv2d(bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.avgpool = nn.Sequential()
        net.fc = nn.Sequential()
        self.net = net
        self.upsample = nn.Upsample(size=(32, 32))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))
        self.fc = nn.Sequential(  # 4608->2
            nn.Linear(512 * 3 * 3, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_class),
        )
        # for name, module in self.net.named_children():
        #     print(name)

    def forward(self, x):
        x_temp = x
        x = self.upsample(x)
        x = self.net(x)
        x = x.reshape([x.shape[0], x.shape[1], 1, 1])
        x = self.avgpool(x)
        if self.mode == 'KD':
            y = x.reshape([-1, 512, 3, 3])  # y的shape 512, 3 ,3 经过了adaptive average pooling
        x = x.view(x.size(0), -1)
        w = x
        x = self.fc(x)
        if self.mode == 'KD':
            if self.srrl == True:
                return x, w
            if self.ret == 'single':
                return x, y
            else:
                z = []
                x_temp = self.upsample(x_temp)
                for name, module in self.net.named_children():
                    x_temp = module(x_temp)
                    if name == 'layer2':
                        z.append(x_temp)
                        break
                z.append(y)
                return x, z
        else:
            return x


# SimKD -- "Knowledge Distillation with the Reused Teacher Classifier" -- CVPR 2022
class SimKD(nn.Module):

    def __init__(self, *, s_n, t_n, factor=2):
        super(SimKD, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False,
                             groups=groups)

        # A bottleneck design to reduce extra parameters
        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n // factor),
            nn.BatchNorm2d(t_n // factor),
            nn.ReLU(inplace=True),
            conv3x3(t_n // factor, t_n // factor),
            # depthwise convolution
            # conv3x3(t_n//factor, t_n//factor, groups=t_n//factor),
            nn.BatchNorm2d(t_n // factor),
            nn.ReLU(inplace=True),
            conv1x1(t_n // factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        ))

    def forward(self, feat_s, feat_t, cls_t):
        # Spatial Dimension Alignment
        s_H, t_H = feat_s.shape[2], feat_t.shape[2]
        # if s_H > t_H:
        #     source = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
        #     target = feat_t
        # else:
        #     source = feat_s
        #     target = F.adaptive_avg_pool2d(feat_t, (s_H, s_H))
        source = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
        target = feat_t

        trans_feat_t = target

        # Channel Alignment
        trans_feat_s = getattr(self, 'transfer')(source)

        # Prediction via Teacher Classifier
        # temp_feat = self.avg_pool(trans_feat_s)
        temp_feat = trans_feat_s.view(trans_feat_s.size(0), -1)
        pred_feat_s = cls_t(temp_feat)

        return trans_feat_s, trans_feat_t, pred_feat_s


# SemCKD -- "Cross-Layer Distillation with Semantic Calibration" -- AAAI 2021
class SemCKDLoss(nn.Module):
    """Cross-Layer Distillation with Semantic Calibration, AAAI2021"""

    def __init__(self):
        super(SemCKDLoss, self).__init__()
        self.crit = nn.MSELoss(reduction='none')

    def forward(self, s_value, f_target, weight):
        bsz, num_stu, num_tea = weight.shape
        ind_loss = torch.zeros(bsz, num_stu, num_tea).cuda()

        for i in range(num_stu):
            for j in range(num_tea):
                ind_loss[:, i, j] = self.crit(s_value[i][j], f_target[i][j]).reshape(bsz, -1).mean(-1)

        loss = (weight * ind_loss).sum() / (1.0 * bsz * num_stu)
        return loss


class SelfA(nn.Module):
    """Cross-layer Self Attention"""

    def __init__(self, feat_dim, s_n, t_n, soft, factor=4):
        super(SelfA, self).__init__()

        self.soft = soft
        self.s_len = len(s_n)
        self.t_len = len(t_n)
        self.feat_dim = feat_dim

        # query and key mapping
        for i in range(self.s_len):
            setattr(self, 'query_' + str(i), MLPEmbed(feat_dim, feat_dim // factor))
        for i in range(self.t_len):
            setattr(self, 'key_' + str(i), MLPEmbed(feat_dim, feat_dim // factor))

        for i in range(self.s_len):
            for j in range(self.t_len):
                setattr(self, 'regressor' + str(i) + str(j), Proj(s_n[i], t_n[j]))

    def forward(self, feat_s, feat_t):
        sim_s = list(range(self.s_len))
        sim_t = list(range(self.t_len))
        bsz = self.feat_dim

        # similarity matrix
        for i in range(self.s_len):
            sim_temp = feat_s[i].reshape(bsz, -1)
            sim_s[i] = torch.matmul(sim_temp, sim_temp.t())
        for i in range(self.t_len):
            sim_temp = feat_t[i].reshape(bsz, -1)
            sim_t[i] = torch.matmul(sim_temp, sim_temp.t())

        # calculate student query
        proj_query = self.query_0(sim_s[0])
        proj_query = proj_query[:, None, :]
        for i in range(1, self.s_len):
            temp_proj_query = getattr(self, 'query_' + str(i))(sim_s[i])
            proj_query = torch.cat([proj_query, temp_proj_query[:, None, :]], 1)

        # calculate teacher key
        proj_key = self.key_0(sim_t[0])
        proj_key = proj_key[:, :, None]
        for i in range(1, self.t_len):
            temp_proj_key = getattr(self, 'key_' + str(i))(sim_t[i])
            proj_key = torch.cat([proj_key, temp_proj_key[:, :, None]], 2)

        # attention weight: batch_size X No. stu feature X No.tea feature
        energy = torch.bmm(proj_query, proj_key) / self.soft
        attention = F.softmax(energy, dim=-1)

        # feature dimension alignment
        proj_value_stu = []
        value_tea = []
        for i in range(self.s_len):
            proj_value_stu.append([])
            value_tea.append([])
            for j in range(self.t_len):
                s_H, t_H = feat_s[i].shape[2], feat_t[j].shape[2]
                if s_H > t_H:
                    source = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))
                    target = feat_t[j]
                elif s_H <= t_H:
                    source = feat_s[i]
                    target = F.adaptive_avg_pool2d(feat_t[j], (s_H, s_H))

                proj_value_stu[i].append(getattr(self, 'regressor' + str(i) + str(j))(source))
                value_tea[i].append(target)

        return proj_value_stu, value_tea, attention


class Proj(nn.Module):
    """feature dimension alignment by 1x1, 3x3, 1x1 convolutions"""

    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(Proj, self).__init__()
        self.num_mid_channel = 2 * num_target_channels

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear mapping for attention calculation"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)
        self.regressor = nn.Sequential(
            nn.Linear(dim_in, 2 * dim_out),
            self.l2norm,
            nn.ReLU(inplace=True),
            nn.Linear(2 * dim_out, dim_out),
            self.l2norm,
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))

        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# SRRL -- "KNOWLEDGE DISTILLATION VIA SOFTMAX REGRESSION REPRESENTATION LEARNING" -- ICLRc 2021
class SRRL(nn.Module):
    """ICLR-2021: Knowledge Distillation via Softmax Regression Representation Learning"""

    def __init__(self, *, s_n, t_n):
        super(SRRL, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        ))

    def forward(self, feat_s, cls_t):
        feat_s = feat_s.unsqueeze(-1).unsqueeze(-1)
        temp_feat = self.transfer(feat_s)
        trans_feat_s = temp_feat.view(temp_feat.size(0), -1)

        pred_feat_s = cls_t(trans_feat_s)

        return trans_feat_s, pred_feat_s


def channel_shuffle(self, x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class bottleblock(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel, stride):
        super(bottleblock, self).__init__()
        self.midchannel = mid_channel
        output = out_channel - in_channel
        self.stride = stride

        self.pointwise_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True))
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, padding=1, stride=stride,
                      groups=mid_channel, bias=False),
            nn.BatchNorm2d(mid_channel))
        self.pointwise_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=mid_channel, out_channels=output, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True))
        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=stride,
                          groups=in_channel, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True))
        else:
            self.shortcut = nn.Sequential()

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

    def forward(self, x):
        if self.stride == 2:
            residual = self.shortcut(x)
            x = self.pointwise_conv1(x)
            x = self.depth_conv(x)
            x = self.pointwise_conv2(x)
            return torch.cat((residual, x), dim=1)
        elif self.stride == 1:
            x1, x2 = self.channel_shuffle(x)
            residual = self.shortcut(x2)
            x1 = self.pointwise_conv1(x1)
            x1 = self.depth_conv(x1)
            x1 = self.pointwise_conv2(x1)
            return torch.cat((residual, x1), dim=1)


class shufflenetv2(nn.Module):
    def __init__(self, bands=3, num_class=2, mode='NKD', ret='single', srrl=False):
        """size表示模型大小"""
        super(shufflenetv2, self).__init__()
        self.mode = mode
        self.ret = ret
        self.srrl = srrl
        net = models.shufflenet_v2_x0_5()
        self.upsample = nn.Upsample(size=(32, 32))
        self.encoder = net
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))
        self.classifier = nn.Sequential(  # 4608->2
            nn.Linear(1000, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_class),
        )
        # for name, module in self.encoder.named_children():
        #     print(name)

    def forward(self, x):
        x_temp = x
        x = self.upsample(x)
        x = self.encoder(x)
        if self.mode == 'KD':
            x = x.view(x.size(0), x.size(1), 1, 1)
            y = self.avgpool(x)
        x = x.view(x.size(0), -1)
        w = x
        x = self.classifier(x)
        if self.mode == 'KD':
            if self.srrl == True:
                return x, w
            if self.ret == 'single':
                return x, y
            else:
                z = []
                x_temp = self.upsample(x_temp)
                for name, module in self.encoder.named_children():
                    x_temp = module(x_temp)
                    if name == 'stage2':
                        z.append(x_temp)
                        break
                z.append(y)
                return x, z
        else:
            return x


class MobileNetv2(nn.Module):
    def __init__(self, bands=3, num_class=2, mode='NKD', ret='single',
                 srrl=False):  # NKD -- not in knowledge distillation mode KD -- in knowledge distillation mode
        # single -- return features output by last conv  all -- return all features
        super(MobileNetv2, self).__init__()
        self.mode = mode
        self.ret = ret
        self.srrl = srrl
        net = models.mobilenet_v2(pretrained=False)
        net.classifier = nn.Sequential()
        # print(net.features)
        from torchvision.ops.misc import Conv2dNormActivation
        net.features[0] = Conv2dNormActivation(3, 32, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        net.features[18] = Conv2dNormActivation(320, 256, stride=2, norm_layer=nn.BatchNorm2d,
                                                activation_layer=nn.ReLU6)
        self.averagepool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        # net.features[18].append(nn.AdaptiveAvgPool2d(output_size=(7,7)))
        self.upsample = nn.Upsample(size=(32, 32))  # the minimum input shape of vgg16 is 32*32
        self.encoder = net
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_class),
        )
        # for name, module in net.features.named_children():
        #     print(name)
        #     print(module)

    def forward(self, x):
        x_temp = x
        x = self.upsample(x)
        x = self.encoder(x)
        if self.mode == 'KD':
            x = x.view(x.size(0), x.size(1), 1, 1)
            y = self.averagepool(x)
        x = x.view(x.size(0), -1)
        w = x
        x = self.classifier(x)
        if self.mode == 'KD':
            if self.srrl == True:
                return x, w
            if self.ret == 'single':
                return x, y
            else:
                z = []
                x_temp = self.upsample(x_temp)
                for name, module in self.encoder.features.named_children():
                    x_temp = module(x_temp)
                    if name == '8':
                        z.append(x_temp)
                        break
                z.append(y)
                return x, z
        else:
            return x


# mobilenet = MobileNetv2(bands=64, num_class=2, mode='KD', ret='single', srrl=False).to("cuda")
# print("mobilenet:\n=================================",mobilenet)
# from torchsummary import summary
# summary(mobilenet, input_size=(64,3,3), device="cuda")
#shufflenet = shufflenetv2(bands=3, num_class=2, mode='NKD', ret='single', srrl=False).to("cuda")
# print("shufflenet:\n=================================",shufflenet)
# model = StudentVGGNet(bands=64, num_class=2).to("cuda")
# summary(model, input_size=(64,3,3))
