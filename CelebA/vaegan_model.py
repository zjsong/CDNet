import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy


# encoder block (used in encoder and discriminator)
class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, ten, out=False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = F.leaky_relu(ten, 0.2, False)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = F.leaky_relu(ten, 0.2, True)
            return ten


# decoder block (used in the decoder)
class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(DecoderBlock, self).__init__()

        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2, output_padding=1,
                                       bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)
        return ten


class Encoder(nn.Module):
    def __init__(self, channel_in=3, z_size=128):
        super(Encoder, self).__init__()
        self.size = channel_in
        layers_list = []

        for i in range(3):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=64))
                self.size = 64
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2

        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=8 * 8 * self.size, out_features=2048, bias=False),
                                nn.BatchNorm1d(num_features=2048, momentum=0.9),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(in_features=2048, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024, momentum=0.9),
                                nn.LeakyReLU(0.2, inplace=True))
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=1024, out_features=z_size)
        self.l_var = nn.Linear(in_features=1024, out_features=z_size)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_size=128, size=256):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=8 * 8 * size, bias=False),
                                nn.BatchNorm1d(num_features=8 * 8 * size, momentum=0.9),
                                nn.ReLU(inplace=True))
        self.size = size
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size))
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size//2))
        self.size = self.size//2
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size//4))
        self.size = self.size//4

        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, 8, 8)
        ten = self.conv(ten)
        return ten


class Discriminator(nn.Module):
    def __init__(self, channel_in=3, recon_level=3):
        super(Discriminator, self).__init__()
        self.size = channel_in
        self.recon_levl = recon_level

        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True)))
        self.size = 32
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=128))
        self.size = 128
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))
        self.size = 256
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))

        self.fc = nn.Sequential(
            nn.Linear(in_features=8 * 8 * self.size, out_features=512, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1)
        )

    def forward(self, ten, other_ten, mode='REC'):
        if mode == "REC":
            ten = torch.cat((ten, other_ten), 0)
            for i, lay in enumerate(self.conv):
                if i == self.recon_levl:
                    ten, layer_ten = lay(ten, True)
                    layer_ten = layer_ten.view(len(layer_ten), -1)
                    return layer_ten
                else:
                    ten = lay(ten)
        else:
            ten = torch.cat((ten, other_ten), 0)
            for i, lay in enumerate(self.conv):
                    ten = lay(ten)

            ten = ten.view(len(ten), -1)
            ten = self.fc(ten)
            return torch.sigmoid(ten)


class VaeGan(nn.Module):
    def __init__(self, device, z_size=128, recon_level=3):
        super(VaeGan, self).__init__()
        self.device = device
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size)
        self.discriminator = Discriminator(channel_in=3, recon_level=recon_level)
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation
                    scale = 1.0/numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                    scale /= numpy.sqrt(3)
                    # nn.init.xavier_normal(m.weight,1)
                    # nn.init.constant(m.weight,0.005)
                    nn.init.uniform(m.weight, -scale, scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, x, z=None, gen_size=10):
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

            dis_output = self.discriminator(x_sampled, x_original, "GAN")

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
    def loss(ten_original, ten_recons, layer_original, layer_recons, labels_original,
             labels_sampled, mus, variances):
        """

        :param ten_original: original images
        :param ten_recons:  reconstructed images (output of the decoder)
        :param layer_original:  intermediate layer for original (intermediate output of the discriminator)
        :param layer_recons: intermediate layer for reconstructed (intermediate output of the discriminator)
        :param labels_original: labels for original (output of the discriminator)
        :param labels_sampled: labels for sampled from gaussian (0,1) (output of the discriminator)
        :param mus: tensor of means
        :param variances: tensor of diagonals of log_variances
        :return:
        """

        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = 0.5*(ten_original.view(len(ten_original), -1) - ten_recons.view(len(ten_recons), -1)) ** 2

        # kl-divergence
        kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1)

        # mse between intermediate layers
        mse = torch.sum(0.5*(layer_original - layer_recons) ** 2, 1)

        # bce for decoder and discriminator for original and sampled
        bce_dis_original = -torch.log(labels_original + 1e-3)
        bce_dis_sampled = -torch.log(1 - labels_sampled + 1e-3)

        bce_gen_sampled = -torch.log(labels_sampled + 1e-3)

        return nle, kl, mse, bce_dis_original, bce_dis_sampled, bce_gen_sampled
