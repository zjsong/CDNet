"""
Learning IcGAN model on MNIST dataset.
"""


import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from icgan_model import cGAN, EncoderZ, EncoderY
from visdom import Visdom
from src.utils import creat_vis_plot, update_vis


def weights_init(model):
    for m in model.parameters():
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


def visualize_results_cgan(params, cgan, device, test_loader):
    cgan.eval()

    n = 10
    plt.figure(figsize=(10, 2))

    with torch.no_grad():
        indices = torch.LongTensor(params.batch_size, 1)
        labels_onehot = torch.FloatTensor(params.batch_size, params.class_num)
        for batch_idx, (batch_x, batch_y) in enumerate(test_loader):

            # convert the labels into one-hot form vectors
            indices.zero_()
            indices = batch_y.view(-1, 1)
            labels_onehot.zero_()
            labels_onehot.scatter_(1, indices, 1)
            batch_y_onehot = labels_onehot.to(device)

            batch_z = torch.randn(batch_y_onehot.size(0), params.dim_z).to(device)
            x_sampled = cgan.Gen(batch_z, batch_y_onehot)

            # display generation
            count = 0
            while count < n:
                count += 1
                ax = plt.subplot(1, n, count)
                image_sampled = x_sampled[count - 1].cpu().numpy().reshape(28, 28)
                plt.imshow(image_sampled)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show(block=False)

            cgan.train()
            break


def visualize_results_encoders(encoder_z, encoder_y, cgan, device, data_loader):
    cgan.eval()
    encoder_z.eval()
    encoder_y.eval()

    n = 10
    plt.figure(figsize=(10, 2))

    with torch.no_grad():
        indices = torch.LongTensor(params.batch_size, 1)
        labels_onehot = torch.FloatTensor(params.batch_size, params.class_num)
        for batch_idx, (batch_x, batch_y) in enumerate(data_loader):

            batch_x = batch_x.to(device)

            # convert the labels into one-hot form vectors
            indices.zero_()
            indices = batch_y.view(-1, 1)
            labels_onehot.zero_()
            labels_onehot.scatter_(1, indices, 1)
            batch_y_onehot = labels_onehot.to(device)

            batch_z = encoder_z(batch_x)

            x_recons_use_y = cgan.Gen(batch_z, batch_y_onehot)

            batch_y_predict = (encoder_y(batch_x) >= 0.5).type(torch.FloatTensor).to(device)
            x_recons_use_y_pred = cgan.Gen(batch_z, batch_y_predict)

            # display reconstructions
            count = 0
            while count < n:
                count += 1

                # originals
                ax = plt.subplot(3, n, count)
                image_original = batch_x[count - 1].cpu().numpy().reshape(28, 28)
                plt.imshow(image_original)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # reconstructions using labels
                ax = plt.subplot(3, n, n + count)
                image_recons_use_y = x_recons_use_y[count - 1].cpu().numpy().reshape(28, 28)
                plt.imshow(image_recons_use_y)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # reconstructions using predicted labels
                ax = plt.subplot(3, n, 2 * n + count)
                image_recons_use_y_pred = x_recons_use_y_pred[count - 1].cpu().numpy().reshape(28, 28)
                plt.imshow(image_recons_use_y_pred)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            plt.show(block=False)

            cgan.train()
            encoder_z.train()
            encoder_y.train()
            break


if __name__ == '__main__':

    # saving path
    save_dir = 'results/icgan'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    parser = argparse.ArgumentParser(description="IcGAN")
    parser.add_argument("--img_sz", type=int, default=28,
                        help="Image size (images have to be squared)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size")
    parser.add_argument("--class_num", type=int, default=10,
                        help="Size of y")
    parser.add_argument("--dim_z", type=int, default=10,
                        help="Size of z")
    parser.add_argument("--input_channel_dis", type=int, default=1,
                        help="No. of the input channel of the 1st conv in Dis")
    parser.add_argument("--n_epochs_cGAN", type=int, default=150,
                        help="Total number of epochs for training cGAN")
    parser.add_argument("--n_epochs_Enc", type=int, default=150,
                        help="Total number of epochs for training EncoderZ and EncoderY")
    parser.add_argument('--n_train', type=int, default=50000,
                        help='The number of training samples')
    parser.add_argument('--n_valid', type=int, default=10000,
                        help='The number of validation samples')
    parser.add_argument('--n_test', type=int, default=10000,
                        help='The number of test samples')
    parser.add_argument("--lr_cGAN", type=float, default=1e-4,
                        help="Learning rate for cGAN")
    parser.add_argument("--lr_EncZ", type=float, default=1e-4,
                        help="Learning rate for EncZ")
    parser.add_argument("--lr_EncY", type=float, default=1e-4,
                        help="Learning rate for EncY")
    parser.add_argument("--decay_lr", type=float, default=0.75,
                        help="Learning rate decay")
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    params = parser.parse_args()

    # use GPUs
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # fix the random seed
    random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)

    # load data
    data_dir = 'D:/Projects/dataset/MNIST'
    train_set = datasets.MNIST(data_dir, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))
    test_set = datasets.MNIST(data_dir, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor()
                              ]))
    # split data
    indices_train_valid = np.arange(len(train_set))
    train_indices = indices_train_valid[:params.n_train]
    valid_indices = indices_train_valid[len(indices_train_valid) - params.n_valid:]
    indices_test = np.arange(len(test_set))
    test_indices = indices_test[:params.n_test]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size,
                                               sampler=SubsetRandomSampler(train_indices))
    valid_loader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size,
                                               sampler=SubsetRandomSampler(valid_indices))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size,
                                              sampler=SubsetRandomSampler(test_indices))

    # build the model
    cgan = cGAN(params, device)
    encoder_z = EncoderZ(params.input_channel_dis, params.dim_z)
    encoder_y = EncoderY(params.input_channel_dis, params.class_num)
    cgan.apply(weights_init)
    encoder_z.apply(weights_init)
    encoder_y.apply(weights_init)
    cgan = cgan.to(device)
    encoder_z = encoder_z.to(device)
    encoder_y = encoder_y.to(device)

    # define optimizers
    optim_gen = optim.Adam(cgan.Gen.parameters(), lr=params.lr_cGAN, betas=(0.5, 0.999))
    optim_dis = optim.Adam(cgan.Dis.parameters(), lr=params.lr_cGAN, betas=(0.5, 0.999))
    optim_enc_z = optim.Adam(encoder_z.parameters(), lr=params.lr_EncZ, betas=(0.5, 0.999))
    optim_enc_y = optim.Adam(encoder_y.parameters(), lr=params.lr_EncY, betas=(0.5, 0.999))

    # schedule learning rate
    Steps = [10000, 20000, 30000, 50000]
    lr_scheduler_gen = MultiStepLR(optim_gen, milestones=Steps, gamma=params.decay_lr)
    lr_scheduler_dis = MultiStepLR(optim_dis, milestones=Steps, gamma=params.decay_lr)
    lr_scheduler_enc_z = MultiStepLR(optim_enc_z, milestones=Steps, gamma=params.decay_lr)
    lr_scheduler_enc_y = MultiStepLR(optim_enc_y, milestones=Steps, gamma=params.decay_lr)

    # define losses
    bce_loss = nn.BCELoss().to(device)
    mse_loss = nn.MSELoss().to(device)

    # start the training
    ################################
    # training cGAN
    ################################
    cgan.train()
    count_update_step_cgan = 0
    indices = torch.LongTensor(params.batch_size, 1)
    labels_onehot = torch.FloatTensor(params.batch_size, params.class_num)

    for i in range(params.n_epochs_cGAN):

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            y_dis_real, y_dis_fake = torch.ones(batch_x.size(0), 1), torch.zeros(batch_x.size(0), 1)
            y_dis_real, y_dis_fake = y_dis_real.to(device), y_dis_fake.to(device)

            batch_x = batch_x.to(device)

            # convert the labels into one-hot form vectors
            indices.zero_()
            indices = batch_y.view(-1, 1)
            labels_onehot.zero_()
            labels_onehot.scatter_(1, indices, 1)
            batch_y_onehot = labels_onehot.to(device)

            # update dis
            for p in cgan.Dis.parameters():
                p.requires_grad_(True)
            for p in cgan.Gen.parameters():
                p.requires_grad_(False)
            optim_dis.zero_grad()
            dis_output_real, dis_output_fake = cgan(batch_y_onehot, batch_x)
            loss_dis_real = bce_loss(dis_output_real, y_dis_real)
            loss_dis_fake = bce_loss(dis_output_fake, y_dis_fake)
            loss_dis = loss_dis_real + loss_dis_fake
            loss_dis.backward()
            optim_dis.step()

            # update gen
            for p in cgan.Gen.parameters():
                p.requires_grad_(True)
            for p in cgan.Dis.parameters():
                p.requires_grad_(False)
            optim_gen.zero_grad()
            _, dis_output_fake = cgan(batch_y_onehot, batch_x)
            loss_gen = bce_loss(dis_output_fake, y_dis_real)
            loss_gen.backward()
            optim_gen.step()

            count_update_step_cgan += 1
            if count_update_step_cgan == 1:
                viz = Visdom()
                x_value = np.asarray(count_update_step_cgan).reshape(1, )
                x_label = 'Training Step'
                y_value = np.column_stack((np.asarray(loss_dis.item()), np.asarray(loss_gen.item())))
                y_label = 'Loss'
                title = 'Discriminator and Generator Losses'
                legend = ['Loss_Dis', 'Loss_Gen']
                win_dis_gen = creat_vis_plot(viz, x_value, y_value,
                                             x_label, y_label, title, legend)
            elif count_update_step_cgan % 50 == 0:
                x_value = np.asarray(count_update_step_cgan).reshape(1, )
                y_value = np.column_stack((np.asarray(loss_dis.item()), np.asarray(loss_gen.item())))
                update_vis(viz, win_dis_gen, x_value, y_value)

            # evaluate the model
            if count_update_step_cgan % 1000 == 0:
                print('\nUpdate step: {:d}'
                      '\nmean loss_dis: {:.4f}'
                      '\nmean loss_gen: {:.4f}'.format(
                    count_update_step_cgan, loss_dis.item(), loss_gen.item()))

            # save the midterm model sates
            if count_update_step_cgan % 5000 == 0:
                torch.save(cgan.state_dict(),
                           save_dir + '/cgan_step' + str(count_update_step_cgan) + '.pt')
                # visualize_results_cgan(params, cgan, device, test_loader)

            lr_scheduler_gen.step()
            lr_scheduler_dis.step()

    # visualize_results_cgan(params, cgan, device, test_loader)
    # save the cgan model
    torch.save(cgan.state_dict(), save_dir + '/cgan_final.pt')

    ################################
    # training EncoderZ and EncoderY
    ################################
    cgan.eval()
    encoder_z.train()
    encoder_y.train()
    count_update_step_enc = 0
    indices = torch.LongTensor(params.batch_size, 1)
    labels_onehot = torch.FloatTensor(params.batch_size, params.class_num)

    for i in range(params.n_epochs_Enc):

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)

            # convert the labels into one-hot form vectors
            indices.zero_()
            indices = batch_y.view(-1, 1)
            labels_onehot.zero_()
            labels_onehot.scatter_(1, indices, 1)
            batch_y_onehot = labels_onehot.to(device)

            # update encoder_z
            optim_enc_z.zero_grad()
            batch_z, batch_x_gen = cgan(batch_y_onehot)
            z_predict = encoder_z(batch_x_gen.detach())
            loss_z = mse_loss(z_predict, batch_z)
            loss_z.backward()
            optim_enc_z.step()

            # update encoder_y
            optim_enc_y.zero_grad()
            y_predict = encoder_y(batch_x)
            loss_y = mse_loss(y_predict, batch_y_onehot)
            loss_y.backward()
            optim_enc_y.step()

            count_update_step_enc += 1
            if count_update_step_enc == 1:
                viz = Visdom()
                x_value = np.asarray(count_update_step_enc).reshape(1, )
                x_label = 'Training Step'

                y_value = np.asarray(loss_z.item()).reshape(1, )
                y_label = 'MSE'
                title = 'Z Reconstruction Loss'
                legend = ['MSE']
                win_z_recons = creat_vis_plot(viz, x_value, y_value,
                                              x_label, y_label, title, legend)

                y_value = np.asarray(loss_y.item()).reshape(1, )
                y_label = 'MSE'
                title = 'Y Reconstruction Loss'
                legend = ['MSE']
                win_y_recons = creat_vis_plot(viz, x_value, y_value,
                                              x_label, y_label, title, legend)
            elif count_update_step_enc % 50 == 0:
                x_value = np.asarray(count_update_step_enc).reshape(1, )

                y_value = np.asarray(loss_z.item()).reshape(1, )
                update_vis(viz, win_z_recons, x_value, y_value)

                y_value = np.asarray(loss_y.item()).reshape(1, )
                update_vis(viz, win_y_recons, x_value, y_value)

            # evaluate the model
            if count_update_step_enc % 1000 == 0:
                print('\nUpdate step: {:d}'
                      '\nmean loss_z: {:.4f}'
                      '\nmean loss_y: {:.4f}'.format(
                    count_update_step_enc, loss_z.item(), loss_y.item()))

            # save the midterm model sates
            if count_update_step_enc % 5000 == 0:
                torch.save(encoder_z.state_dict(),
                           save_dir + '/encoder_z_step' + str(count_update_step_enc) + '.pt')
                torch.save(encoder_y.state_dict(),
                           save_dir + '/encoder_y_step' + str(count_update_step_enc) + '.pt')
                # visualize_results_encoders(encoder_z, encoder_y, cgan, device, test_loader)

            lr_scheduler_enc_z.step()
            lr_scheduler_enc_y.step()

    # visualize_results_encoders(encoder_z, encoder_y, cgan, device, test_loader)
    # save two encoder models
    torch.save(encoder_z.state_dict(), save_dir + '/encoder_z_final.pt')
    torch.save(encoder_y.state_dict(), save_dir + '/encoder_y_final.pt')

    # save the settings and results
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write('Settings and Results:\n')
        f.write('------------------------------\n')
        f.write('Random seed = ' + str(params.seed) + '\n')
        f.write('Image size = ' + str(params.img_sz) + '\n')
        f.write('Batch size = ' + str(params.batch_size) + '\n')
        f.write('Learning rate of cGAN = ' + str(params.lr_cGAN) + '\n')
        f.write('Learning rate of EncoderZ = ' + str(params.lr_EncZ) + '\n')
        f.write('Learning rate of EncoderY = ' + str(params.lr_EncY) + '\n')
        f.write('Learning rate decay = ' + str(params.decay_lr) + '\n')
        f.write('------------------------------\n')
        f.write('Training samples = ' + str(params.n_train) + '\n')
        f.write('Validation samples = ' + str(params.n_valid) + '\n')
        f.write('Test samples = ' + str(params.n_test) + '\n')
        f.write('------------------------------\n')
        f.write('Class (or Attribute) vector size = ' + str(params.class_num) + '\n')
        f.write('Latent vector size = ' + str(params.dim_z) + '\n')
        f.write('No. of the input channel of the 1st conv in Dis = ' + str(params.input_channel_dis) + '\n')
        f.write('------------------------------\n')
        f.write('Max epoch for cGAN = ' + str(params.n_epochs_cGAN) + '\n')
        f.write('Max epoch for EncoderZ and EncoderY = ' + str(params.n_epochs_Enc) + '\n')
        f.write('Total update steps for cGAN = ' + str(count_update_step_cgan) + '\n')
        f.write('Total update steps for EncoderZ and EncoderY = ' + str(count_update_step_enc) + '\n')









