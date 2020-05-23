"""
Generator, Discriminator, Encoder_z, and Encoder_y.
"""


import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, dim_z=10, class_num=10):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(dim_z + class_num, 7 * 7 * 64),
            nn.BatchNorm1d(7 * 7 * 64),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, latent, label):
        z = torch.cat((latent, label), 1)
        z = self.fc(z)
        z = z.view(len(z), -1, 7, 7)
        x = self.deconv(z)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channel=1, class_num=10):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, 5, 1, 2),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_main = nn.Sequential(
            nn.Conv2d(64 + class_num, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        z = self.conv1(x)

        y = label.unsqueeze(2).unsqueeze(3)
        y = y.expand(y.size(0), y.size(1), z.size(2), z.size(3))
        z_y = torch.cat((z, y), 1)
        z = self.conv_main(z_y)

        z = z.view(len(z), -1)
        z = self.fc(z)
        return z


class cGAN(nn.Module):
    def __init__(self, params, device):
        super(cGAN, self).__init__()
        self.device = device
        self.dim_z = params.dim_z
        self.class_num = params.class_num
        self.input_channel_dis = params.input_channel_dis

        self.Gen = Generator(self.dim_z, self.class_num)
        self.Dis = Discriminator(self.input_channel_dis, self.class_num)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation
                    scale = 1.0 / np.sqrt(np.prod(m.weight.shape[1:]))
                    scale /= np.sqrt(3)
                    # nn.init.xavier_normal(m.weight,1)
                    # nn.init.constant(m.weight,0.005)
                    nn.init.uniform(m.weight, -scale, scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, y, x=None):
        if x is not None:
            # for real data
            dis_output_real = self.Dis(x, y)

            # for fake data
            z = torch.randn(y.size(0), self.dim_z).to(self.device)
            x_fake = self.Gen(z, y)
            dis_output_fake = self.Dis(x_fake, y)

            return dis_output_real, dis_output_fake
        else:
            z = torch.randn(y.size(0), self.dim_z).to(self.device)
            x_gen = self.Gen(z, y)
            return z, x_gen


class EncoderZ(nn.Module):
    def __init__(self, input_channel=1, dim_output=10):
        super(EncoderZ, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, dim_output)
        )

    def forward(self, x):
        z = self.conv(x)
        z = z.view(len(z), -1)
        out = self.fc(z)
        return out


class EncoderY(nn.Module):
    def __init__(self, input_channel=1, class_num=10):
        super(EncoderY, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, class_num)
        )

    def forward(self, x):
        z = self.conv(x)
        z = z.view(len(z), -1)
        out = self.fc(z)
        return out
