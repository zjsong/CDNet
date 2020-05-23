"""
Encoder and Decoder.
"""


import torch
import torch.nn.functional as F
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                              kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out)

    def forward(self, x, out=False):
        if out:
            z = self.conv(x)
            mid_out = z
            z = self.bn(z)
            y = F.leaky_relu(z, 0.2, True)
            return y, mid_out
        else:
            z = self.conv(x)
            z = self.bn(z)
            y = F.leaky_relu(z, 0.2, True)
            return y


class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(channel_in, channel_out,
                                       kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out)

    def forward(self, mix_z):
        z = self.conv(mix_z)
        z = self.bn(z)
        x = F.relu(z, True)
        return x


class Encoder(nn.Module):
    def __init__(self, channel_in=1, num_classe=10, latent_size=10):
        super(Encoder, self).__init__()
        self.size = channel_in
        self.num_classe = num_classe
        layers_list = []

        for i in range(2):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=32))
                self.size = 32
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2

        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=7 * 7 * self.size, out_features=1000, bias=False),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Dropout(p=0.25),
                                nn.Linear(in_features=1000, out_features=num_classe + latent_size))
    
    def forward(self, x):
        z = self.conv(x)
        z = z.view(len(z), -1)
        z = self.fc(z)
        y_class = z[:, :self.num_classe]
        z_latent = z[:, self.num_classe:]
        return y_class, z_latent


class Decoder(nn.Module):
    def __init__(self, z_size=20, channel_num=64):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=1000, bias=False),
                                nn.BatchNorm1d(num_features=1000),
                                nn.ReLU(True),
                                nn.Linear(in_features=1000, out_features=7 * 7 * channel_num, bias=False),
                                nn.BatchNorm1d(num_features=7 * 7 * channel_num),
                                nn.ReLU(True))
        self.channel_num = channel_num
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=self.channel_num, channel_out=self.channel_num//2))
        self.channel_num = self.channel_num // 2
        layers_list.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.channel_num, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, label_class, z_latent):
        z = torch.cat((label_class, z_latent), 1)
        z = self.fc(z)
        z = z.view(len(z), -1, 7, 7)
        x = self.conv(z)
        return x
