"""
Encoder_Z, Encoder_Y, Decoder/Generator, and Discriminator.
"""


import torch
import torch.nn.functional as F
import torch.nn as nn


# encoder block (used in encoder and discriminator)
class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                              kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out)

    def forward(self, x, out=False):
        # here we want to be able to take an intermediate output for reconstruction error
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


# decoder block (used in the decoder)
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


class Encoder_Y(nn.Module):
    def __init__(self, channel_in=1, num_classe=10):
        super(Encoder_Y, self).__init__()
        self.channel_num = channel_in
        layers_list = []

        for i in range(2):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.channel_num, channel_out=32))
                self.channel_num = 32
            else:
                layers_list.append(EncoderBlock(channel_in=self.channel_num, channel_out=self.channel_num * 2))
                self.channel_num *= 2

        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=7 * 7 * self.channel_num, out_features=1000, bias=False),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(in_features=1000, out_features=num_classe))

    def forward(self, x):
        z = self.conv(x)
        z = z.view(len(z), -1)
        y_class = self.fc(z)
        return y_class


class Encoder_Z(nn.Module):
    def __init__(self, channel_in=1, latent_size=10):
        super(Encoder_Z, self).__init__()
        self.channel_num = channel_in
        layers_list = []

        for i in range(2):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.channel_num, channel_out=32))
                self.channel_num = 32
            else:
                layers_list.append(EncoderBlock(channel_in=self.channel_num, channel_out=self.channel_num * 2))
                self.channel_num *= 2

        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=7 * 7 * self.channel_num, out_features=1000, bias=False),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(in_features=1000, out_features=latent_size))

    def forward(self, x):
        z = self.conv(x)
        z = z.view(len(z), -1)
        z_latent = self.fc(z)
        return z_latent


class Decoder(nn.Module):
    def __init__(self, z_size=20, channel_num=64):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(z_size, 1000, bias=False)
        self.fc1_bn = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1010, 7 * 7 * channel_num, bias=False)
        self.fc2_bn = nn.BatchNorm1d(7 * 7 * channel_num)

        self.deconv1 = nn.ConvTranspose2d(channel_num + 10, channel_num // 2,
                                          kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv1_bn = nn.BatchNorm2d(channel_num // 2)

        self.deconv2 = nn.ConvTranspose2d(channel_num // 2 + 10, 1,
                                          kernel_size=4, stride=2, padding=1)

    def forward(self, z_latent, y_class_softmax):
        z_y = torch.cat((z_latent, y_class_softmax), 1)
        z = F.relu(self.fc1_bn(self.fc1(z_y)), True)

        z_y = torch.cat((z, y_class_softmax), 1)
        z = F.relu(self.fc2_bn(self.fc2(z_y)), True)

        z = z.view(len(z), -1, 7, 7)
        y = y_class_softmax.unsqueeze(2).unsqueeze(3)
        y_feature_map = y.expand(y.size(0), 10, z.size(2), z.size(2))
        z_y = torch.cat((z, y_feature_map), 1)
        z = F.relu(self.deconv1_bn(self.deconv1(z_y)), True)

        y_feature_map = y.expand(y.size(0), 10, z.size(2), z.size(2))
        z_y = torch.cat((z, y_feature_map), 1)
        x = torch.sigmoid(self.deconv2(z_y))
        return x


class Discriminator(nn.Module):
    def __init__(self, channel_in=1, recon_level=2):
        super(Discriminator, self).__init__()
        self.recon_levl = recon_level

        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True)))
        self.channel_num = 32
        self.conv.append(EncoderBlock(channel_in=self.channel_num, channel_out=64))
        self.channel_num = 64
        self.conv.append(EncoderBlock(channel_in=self.channel_num, channel_out=128))
        self.channel_num = 128

        self.fc = nn.Sequential(
            nn.Linear(in_features=7 * 7 * self.channel_num, out_features=128, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=1))

    def forward(self, x_rec, x_original, mode='REC'):
        z = torch.cat((x_rec, x_original), 0)
        if mode == "REC":
            for i, lay in enumerate(self.conv):
                if i == self.recon_levl:
                    z, layer_repre = lay(z, True)
                    layer_repre = layer_repre.view(len(layer_repre), -1)
                    return layer_repre
                else:
                    z = lay(z)
        else:
            for i, lay in enumerate(self.conv):
                z = lay(z)

            z = z.view(len(z), -1)
            output = self.fc(z)
            return torch.sigmoid(output)
