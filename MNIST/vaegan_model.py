"""
Encoder, Decoder/Generator, and Discriminator.
"""


import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


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


class Encoder(nn.Module):
    def __init__(self, channel_in=1, z_size=20):
        super(Encoder, self).__init__()
        self.size = channel_in
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
                                nn.BatchNorm1d(num_features=1000),
                                nn.LeakyReLU(0.2, inplace=True))
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=1000, out_features=z_size)
        self.l_var = nn.Linear(in_features=1000, out_features=z_size)

    def forward(self, x):
        z = self.conv(x)
        z = z.view(len(z), -1)
        z = self.fc(z)
        mu = self.l_mu(z)
        logvar = self.l_var(z)
        return mu, logvar


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
        self.channel_num = self.channel_num//2

        layers_list.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.channel_num, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(len(z), -1, 7, 7)
        x = self.conv(z)
        return x


class Discriminator(nn.Module):
    def __init__(self, channel_in=1, recon_level=2):
        super(Discriminator, self).__init__()
        self.channel_num = channel_in
        self.recon_levl = recon_level

        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True)))
        self.channel_num = 32
        self.conv.append(EncoderBlock(channel_in=self.channel_num, channel_out=64))
        self.channel_num = 64
        self.conv.append(EncoderBlock(channel_in=self.channel_num, channel_out=128))
        self.channel_num = 128

        self.fc = nn.Sequential(
            nn.Linear(in_features=7 * 7 * self.channel_num, out_features=128, bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),
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


class VaeGan(nn.Module):
    def __init__(self, device, z_size=20, recon_level=2):
        super(VaeGan, self).__init__()
        self.device = device
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size)
        self.decoder = Decoder(z_size=self.z_size, channel_num=64)
        self.discriminator = Discriminator(channel_in=1, recon_level=recon_level)
        self.init_parameters()

    def init_parameters(self):
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

    def forward(self, x, z=None, gen_size=20):
        if self.training:
            # save the original images
            x_original = x

            # encode
            mus, log_variances = self.encoder(x)
            variances = torch.exp(log_variances * 0.5)
            ten_from_normal = torch.randn(len(x), self.z_size).to(self.device)
            ten_from_normal.requires_grad = True
            z = ten_from_normal * variances + mus

            # decode for reconstructions
            x_recons = self.decoder(z)

            # discriminator for reconstruction
            mid_repre = self.discriminator(x_recons, x_original, "REC")

            # decoder for samples
            ten_from_normal = torch.randn(len(x), self.z_size).to(self.device)
            ten_from_normal.requires_grad = True
            x_sampled = self.decoder(ten_from_normal)

            dis_output = self.discriminator(x_original, x_sampled, "GAN")

            return x_recons, dis_output, mid_repre, mus, log_variances

        else:
            if x is None:
                # just sample and decode
                z = torch.randn(gen_size, self.z_size).to(self.device)
                x_gen = self.decoder(z)
            else:
                if z is None:
                    mus, log_variances = self.encoder(x)
                    variances = torch.exp(log_variances * 0.5)
                    ten_from_normal = torch.randn(len(x), self.z_size).to(self.device)
                    z = ten_from_normal * variances + mus
                    # decode the tensor
                    x_gen = self.decoder(z)
                else:
                    mus, log_variances = self.encoder(x)
                    variances = torch.exp(log_variances * 0.5)
                    ten_from_normal = torch.randn(len(x), self.z_size).to(self.device)
                    z_original = ten_from_normal * variances + mus
                    z += z_original
                    x_gen = self.decoder(z)
            return x_gen, z

    @staticmethod
    def loss(x_original, x_predict, layer_original, layer_predicted, labels_original,
             labels_sampled, mus, variances):
        """

        :param x_original: original images
        :param x_predict:  predicted images (output of the decoder)
        :param layer_original:  intermediate layer for original (intermediate output of the discriminator)
        :param layer_predicted: intermediate layer for reconstructed (intermediate output of the discriminator)
        :param labels_original: labels for original (output of the discriminator)
        :param labels_predicted: labels for reconstructed (output of the discriminator)
        :param labels_sampled: labels for sampled from gaussian (0,1) (output of the discriminator)
        :param mus: tensor of means
        :param variances: tensor of diagonals of log_variances
        :return:
        """

        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = 0.5*(x_original.view(len(x_original), -1) - x_predict.view(len(x_predict), -1)) ** 2

        # kl-divergence
        kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1)

        # mse between intermediate layers
        mse = torch.sum(0.5*(layer_original - layer_predicted) ** 2, 1)

        # bce for decoder and discriminator
        dis_original = -torch.log(labels_original + 1e-3)
        dis_sampled = -torch.log(1 - labels_sampled + 1e-3)

        # bce_gen_original = -torch.log(1-labels_original + 1e-3)
        # bce_gen_sampled = -torch.log(labels_sampled + 1e-3)

        return nle, kl, mse, dis_original, dis_sampled
