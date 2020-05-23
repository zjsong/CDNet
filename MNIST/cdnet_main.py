"""
CDNet model for learning controllable disentangled representations on MNIST dataset.
"""


import os
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from visdom import Visdom
from cdnet_model import Encoder_Y, Encoder_Z, Decoder, Discriminator
from src.cross_covariance import XCov
from src.distance_covariance import dCov2
from src.utils import get_lambda, creat_vis_plot, update_vis


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


# model training function
def training(params, encoder_y, encoder_z, decoder, discriminator,
             optim_encoder_y, optim_encoder_z, optim_decoder, optim_discriminator,
             lr_scheduler_encoder_y, lr_scheduler_encoder_z, lr_scheduler_decoder, lr_scheduler_dis,
             device, digit_class, decorr_regul, train_loader, valid_loader,
             margin, equilibrium, save_dir):

    ################################################
    # training Encoder_Y
    ################################################
    encoder_y.train()
    count_update_step_class = 0

    for i in range(params.n_epochs_EncY):

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optim_encoder_y.zero_grad()
            y_class = encoder_y(batch_x)
            loss_class = digit_class(y_class, batch_y)
            loss_class.backward()
            optim_encoder_y.step()

            count_update_step_class += 1
            if count_update_step_class % 500 == 0:
                # evaluation on validation set
                error_rate = evaluate_class(encoder_y, valid_loader, params.n_valid, device)
                print('\nUpdate step: {:d} '
                      '\nClassification Error Rate on Validation Set: {:.4f}%'.format(
                    count_update_step_class, error_rate))

                if count_update_step_class == 500:
                    viz = Visdom()
                    x_value = np.asarray(count_update_step_class).reshape(1, )
                    x_label = 'Training Step'
                    y_value = np.asarray(error_rate).reshape(1, )
                    y_label = 'Classification Error'
                    title = 'Valid Attr. Classification Error'
                    legend = ['Class_Error']
                    win_digit_class_valid = creat_vis_plot(viz, x_value, y_value,
                                                           x_label, y_label, title, legend)
                else:
                    x_value = np.asarray(count_update_step_class).reshape(1, )
                    y_value = np.asarray(error_rate).reshape(1, )
                    update_vis(viz, win_digit_class_valid, x_value, y_value)

                encoder_y.train()

            # save the midterm model sate
            if count_update_step_class % 1000 == 0:
                torch.save(encoder_y.state_dict(), save_dir + '/encoder_y_step' + str(count_update_step_class) + '.pt')

            lr_scheduler_encoder_y.step()

    error_rate = evaluate_class(encoder_y, valid_loader, params.n_valid, device)
    print('\nFinal Classification Error Rate on Validation Set: {:.4f}%'.format(error_rate))
    torch.save(encoder_y.state_dict(), save_dir + '/encoder_y_final.pt')

    ################################################
    # training Encoder_Z, Decoder, and Discriminator
    ################################################
    encoder_y.eval()
    encoder_z.train()
    decoder.train()
    discriminator.train()

    count_update_step = 0
    for i in range(params.n_epochs):

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):

            batch_x = batch_x.to(device)

            count_update_step += 1

            ############################
            # (1) update discriminator
            ############################
            for p in discriminator.parameters():
                p.requires_grad_(True)
            for p in decoder.parameters():
                p.requires_grad_(False)
            for p in encoder_z.parameters():
                p.requires_grad_(False)
            optim_discriminator.zero_grad()

            y_class = encoder_y(batch_x)
            y_class_softmax = F.softmax(y_class, 1)
            z_latent = encoder_z(batch_x)
            x_recons = decoder(z_latent.detach(), y_class_softmax)
            # using x_recons as fake data
            dis_output = discriminator(x_recons.detach(), batch_x, mode='GAN')
            dis_output_sampled = dis_output[:batch_x.size(0)]
            dis_output_original = dis_output[batch_x.size(0):]

            # GAN loss
            dis_original = -torch.log(dis_output_original + 1e-3)
            dis_sampled = -torch.log(1 - dis_output_sampled + 1e-3)

            loss_discriminator = torch.mean(dis_original) + torch.mean(dis_sampled)

            train_dis = True
            train_dec = True
            if ((torch.mean(dis_original)).item() > equilibrium + margin) \
                    or ((torch.mean(dis_sampled)).item() > equilibrium + margin):
                train_dec = False
            if ((torch.mean(dis_original)).item() < equilibrium - margin) \
                    or ((torch.mean(dis_sampled)).item() < equilibrium - margin):
                train_dis = False
            if train_dec is False and train_dis is False:
                train_dis = True
                train_dec = True
            if train_dis:
                loss_discriminator.backward()
                optim_discriminator.step()

            ############################
            # (2) update decoder
            ############################
            for p in decoder.parameters():
                p.requires_grad_(True)
            for p in discriminator.parameters():
                p.requires_grad_(False)
            optim_decoder.zero_grad()

            y_class = encoder_y(batch_x)
            y_class_softmax = F.softmax(y_class, 1)
            z_latent = encoder_z(batch_x)
            x_recons = decoder(z_latent.detach(), y_class_softmax)
            mid_repre = discriminator(x_recons, batch_x, mode='REC')
            mid_repre_recons = mid_repre[:batch_x.size(0)]
            mid_repre_original = mid_repre[batch_x.size(0):]
            # using x_recons as fake data
            dis_output = discriminator(x_recons, batch_x, mode='GAN')
            dis_output_sampled = dis_output[:batch_x.size(0)]
            dis_output_original = dis_output[batch_x.size(0):]

            # image reconstruction loss
            loss_recons_image = torch.mean(0.5 * (batch_x.view(len(batch_x), -1)
                                                  - x_recons.view(len(x_recons), -1)) ** 2)
            # feature reconstruction loss
            loss_recons_feature = torch.mean(0.5 * (mid_repre_original - mid_repre_recons) ** 2)
            # GAN loss
            dis_original = -torch.log(dis_output_original + 1e-3)
            dis_sampled = -torch.log(1 - dis_output_sampled + 1e-3)
            loss_discriminator = torch.mean(dis_original) + torch.mean(dis_sampled)

            loss_decoder = loss_recons_image + params.lambda_recons * loss_recons_feature - \
                           params.lambda_dis * loss_discriminator

            train_dis = True
            train_dec = True
            if ((torch.mean(dis_original)).item() > equilibrium + margin) \
                    or ((torch.mean(dis_sampled)).item() > equilibrium + margin):
                train_dec = False
            if ((torch.mean(dis_original)).item() < equilibrium - margin) \
                    or ((torch.mean(dis_sampled)).item() < equilibrium - margin):
                train_dis = False
            if train_dec is False and train_dis is False:
                train_dis = True
                train_dec = True
            if train_dec:
                loss_decoder.backward()
                optim_decoder.step()

            ############################
            # (3) update encoder_z
            ############################
            for p in encoder_z.parameters():
                p.requires_grad_(True)
            for p in decoder.parameters():
                p.requires_grad_(False)
            optim_encoder_z.zero_grad()

            y_class = encoder_y(batch_x)
            y_class_softmax = F.softmax(y_class, 1)
            z_latent = encoder_z(batch_x)
            x_recons = decoder(z_latent, y_class_softmax)
            mid_repre = discriminator(x_recons, batch_x, mode='REC')
            mid_repre_recons = mid_repre[:batch_x.size(0)]
            mid_repre_original = mid_repre[batch_x.size(0):]

            # decorrelation loss
            start_time = time.time()
            loss_decorr = decorr_regul(y_class_softmax, z_latent)
            end_time = time.time()
            print('Time cost of computing decorr_regul: batch_id={:d}, time={:.9f}'.
                  format(batch_idx, end_time - start_time))
            # image reconstruction loss
            loss_recons_image = torch.mean(0.5 * (batch_x.view(len(batch_x), -1)
                                                  - x_recons.view(len(x_recons), -1)) ** 2)
            # feature reconstruction loss
            loss_recons_feature = torch.mean(0.5 * (mid_repre_original - mid_repre_recons) ** 2)

            loss_encoder_z = loss_recons_image + params.lambda_recons * loss_recons_feature + \
                             get_lambda(params.lambda_decorr, params.lambda_schedule, count_update_step) * loss_decorr

            loss_encoder_z.backward()
            optim_encoder_z.step()

            # visualize losses
            if count_update_step == 1:
                viz = Visdom()
                x_value = np.asarray(count_update_step).reshape(1, )
                x_label = 'Training Step'

                y_value = np.asarray(loss_recons_image.item()).reshape(1, )
                y_label = 'MSE'
                title = 'Image Reconstruction Loss'
                legend = ['MSE']
                win_img_recons = creat_vis_plot(viz, x_value, y_value,
                                                x_label, y_label, title, legend)

                y_value = np.asarray(loss_recons_feature.item()).reshape(1, )
                y_label = 'MSE'
                title = 'Feature Reconstruction Loss'
                legend = ['MSE']
                win_feature_recons = creat_vis_plot(viz, x_value, y_value,
                                                    x_label, y_label, title, legend)

                y_value = np.column_stack((np.asarray(loss_discriminator.item()), np.asarray(loss_decoder.item())))
                y_label = 'Loss'
                title = 'Discriminator and Decoder Losses'
                legend = ['Loss_Dis', 'Loss_Dec']
                win_dis_gen = creat_vis_plot(viz, x_value, y_value,
                                             x_label, y_label, title, legend)

                y_value = np.asarray(loss_decorr.item()).reshape(1, )
                y_label = 'Loss'
                title = 'Decorrelation Loss'
                legend = ['Loss_Decorr']
                win_decorr = creat_vis_plot(viz, x_value, y_value,
                                            x_label, y_label, title, legend)

            elif count_update_step % 50 == 0:
                x_value = np.asarray(count_update_step).reshape(1, )

                y_value = np.asarray(loss_recons_image.item()).reshape(1, )
                update_vis(viz, win_img_recons, x_value, y_value)

                y_value = np.asarray(loss_recons_feature.item()).reshape(1, )
                update_vis(viz, win_feature_recons, x_value, y_value)

                y_value = np.column_stack((np.asarray(loss_discriminator.item()), np.asarray(loss_decoder.item())))
                update_vis(viz, win_dis_gen, x_value, y_value)

                y_value = np.asarray(loss_decorr.item()).reshape(1, )
                update_vis(viz, win_decorr, x_value, y_value)

            # evaluate the model
            if count_update_step % 1000 == 0:
                print('\nUpdate step: {:d}'
                      '\nmean loss_img_recons: {:.4f}'
                      '\nmean loss_feature_recons: {:.4f}'
                      '\nmean loss_GAN_dis: {:.4f}'
                      '\nmean loss_decoder: {:.4f}'
                      '\nmean loss_decorr: {:.4f}'.format(
                    count_update_step, loss_recons_image.item(), loss_recons_feature.item(),
                    loss_discriminator.item(), loss_decoder.item(), loss_decorr.item()))

                # evaluation on validation set
                loss_decorr_valid, loss_recons_valid = evaluate_learning(encoder_y, encoder_z, decoder, valid_loader,
                                                                         params.n_valid, decorr_regul, device)
                print('Decorrelation Error on Each Mini-Batch: {:.4f}'.format(loss_decorr_valid))
                print('Reconstruction Error on Each Sample: {:.4f}'.format(loss_recons_valid))

                if count_update_step == 1000:
                    viz = Visdom()
                    x_value = np.asarray(count_update_step).reshape(1, )
                    x_label = 'Training Step'

                    y_value = np.asarray(loss_recons_valid).reshape(1, )
                    y_label = 'Reconstruction Error'
                    title = 'Valid Image Reconstruction Error'
                    legend = ['Recons_Error']
                    win_img_recons_valid = creat_vis_plot(viz, x_value, y_value,
                                                          x_label, y_label, title, legend)

                    y_value = np.asarray(loss_decorr_valid).reshape(1, )
                    y_label = 'Decorrelation Error'
                    title = 'Valid Decorrelation Error'
                    legend = ['Decorr_Error']
                    win_decorr_valid = creat_vis_plot(viz, x_value, y_value,
                                                      x_label, y_label, title, legend)
                else:
                    x_value = np.asarray(count_update_step).reshape(1, )

                    y_value = np.asarray(loss_recons_valid).reshape(1, )
                    update_vis(viz, win_img_recons_valid, x_value, y_value)

                    y_value = np.asarray(loss_decorr_valid).reshape(1, )
                    update_vis(viz, win_decorr_valid, x_value, y_value)

                encoder_z.train()
                decoder.train()

            # save the midterm model sates
            if count_update_step % 5000 == 0:
                torch.save(encoder_z.state_dict(),
                           save_dir + '/encoder_z_step' + str(count_update_step) + '.pt')
                torch.save(decoder.state_dict(),
                           save_dir + '/decoder_step' + str(count_update_step) + '.pt')
                torch.save(discriminator.state_dict(),
                           save_dir + '/discriminator_step' + str(count_update_step) + '.pt')

            lr_scheduler_encoder_z.step()
            lr_scheduler_decoder.step()
            lr_scheduler_dis.step()

    torch.save(encoder_z.state_dict(), save_dir + '/encoder_z_final.pt')
    torch.save(decoder.state_dict(), save_dir + '/decoder_final.pt')
    torch.save(discriminator.state_dict(), save_dir + '/discriminator_final.pt')

    return count_update_step


# test function
def evaluate_class(encoder_y, data_loader, num_data, device):
    encoder_y.eval()

    correct_class = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_class = encoder_y(batch_x)
            pred = y_class.max(1, keepdim=True)[1]
            correct_class += pred.eq(batch_y.view_as(pred)).cpu().sum().item()

    error_rate = 100 * (1 - correct_class / num_data)

    return error_rate


# test function
def evaluate_learning(encoder_y, encoder_z, decoder, data_loader, num_data, decorr_regul, device):
    encoder_z.eval()
    decoder.eval()

    loss_decorr = 0
    loss_recons = 0
    num_batch = 0
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(data_loader):

            batch_x = batch_x.to(device)
            num_batch += 1

            y_class = encoder_y(batch_x)
            y_class_softmax = F.softmax(y_class, 1)
            z_latent = encoder_z(batch_x)
            x_recons = decoder(z_latent, y_class_softmax)

            # decorrelation
            loss_decorr += decorr_regul(y_class_softmax, z_latent).item()
            # reconstruction
            loss_recons += torch.sum(torch.mean(0.5 * (batch_x.view(len(batch_x), -1)
                                                       - x_recons.view(len(x_recons), -1)) ** 2, 1)).item()
    loss_decorr /= num_batch
    loss_recons /= num_data

    return loss_decorr, loss_recons


# visualize the reconstruction results
def visual_recons(encoder_y, encoder_z, decoder, device, test_loader):
    encoder_z.eval()
    decoder.eval()

    n = 5
    plt.figure(figsize=(10, 4))
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(test_loader):

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_class = encoder_y(batch_x)
            y_class_softmax = F.softmax(y_class, 1)
            z_latent = encoder_z(batch_x)
            outputs = decoder(z_latent, y_class_softmax)

            in_out = zip(batch_x, outputs)
            count = 0
            for image, output in in_out:
                count += 1
                if count > n:
                    break

                # display original images
                ax = plt.subplot(2, n, count)
                image = image.cpu().numpy().reshape(28, 28)
                plt.imshow(image)
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # display reconstruction
                ax = plt.subplot(2, n, n + count)
                image_recons = output.cpu().numpy().reshape(28, 28)
                plt.imshow(image_recons)
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            plt.show(block=False)

            # fix latent representation, and change attribute label
            plt.figure(figsize=(3, 1))
            # original image
            subject = 0
            image_test = batch_x[subject].cpu().numpy().reshape(28, 28)
            z_test = z_latent[subject].view(1, -1)
            label_original = batch_y[subject].view(1, -1)
            print(label_original)
            y_class_original = y_class_softmax[subject, :].view(1, -1)
            print(y_class_original)
            ax = plt.subplot(1, 3, 1)
            plt.imshow(image_test)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # reconstructed image
            output_test = decoder(z_test, y_class_original)
            output_test = output_test[0].cpu().numpy().reshape(28, 28)
            ax = plt.subplot(1, 3, 2)
            plt.imshow(output_test)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # manipulating image
            curr_digit_id = batch_y[subject].item()
            desir_digit_id = 5
            label_test = y_class_original.clone().cpu().numpy()
            curr_digit_value = label_test[0, curr_digit_id]
            desir_digit_value = label_test[0, desir_digit_id]
            label_test[0, curr_digit_id] = desir_digit_value
            label_test[0, desir_digit_id] = curr_digit_value
            label_test = torch.from_numpy(label_test).to(device)
            output_test = decoder(z_test, label_test)
            output_test = output_test.cpu().detach().numpy().reshape(28, 28)
            ax = plt.subplot(1, 3, 3)
            plt.imshow(output_test)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.show(block=False)

            break


def main():
    # saving path
    save_dir = 'results/cdnet_dcov'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # parse parameters
    parser = argparse.ArgumentParser(description='CDNet')
    parser.add_argument('--img_sz', type=int, default=28,
                        help='Image size (images have to be squared)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size')
    parser.add_argument('--n_class', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--latent_size', type=int, default=10,
                        help='Latent vector size')
    parser.add_argument('--recon_level', type=int, default=2,
                        help='Level wherein computing feature reconstruction error')
    parser.add_argument('--n_epochs_EncY', type=int, default=50,
                        help='Total number of epochs for training Encoder_Y')
    parser.add_argument('--n_epochs', type=int, default=250,
                        help='Total number of epochs')
    parser.add_argument('--n_train', type=int, default=50000,
                        help='The number of training samples')
    parser.add_argument('--n_valid', type=int, default=10000,
                        help='The number of validation samples')
    parser.add_argument('--n_test', type=int, default=10000,
                        help='The number of test samples')
    parser.add_argument('--lr_encoder_y', type=float, default=1e-4,
                        help='Learning rate for encoder_y')
    parser.add_argument('--lr_encoder_z', type=float, default=1e-4,
                        help='Learning rate for encoder_z')
    parser.add_argument('--lr_decoder', type=float, default=1e-4,
                        help='Learning rate for decoder')
    parser.add_argument('--lr_discriminator', type=float, default=1e-4,
                        help='Learning rate for discriminator')
    parser.add_argument('--lambda_decorr', type=float, default=1,
                        help='Decorrelation regularization coefficient')
    parser.add_argument("--lambda_schedule", type=float, default=50000,
                        help="Progressively increase decorrelation lambda (0 to disable)")
    parser.add_argument('--lambda_recons', type=float, default=1,
                        help='Feature reconstruction coefficient for decoder')
    parser.add_argument('--lambda_dis', type=float, default=1e-2,
                        help='Discriminator coefficient')
    parser.add_argument('--decay_lr', type=float, default=0.75,
                        help='Learning rate decay')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    params = parser.parse_args()

    # margin and equilibirum
    margin = 0.35
    equilibrium = 0.68

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
    encoder_y = Encoder_Y(channel_in=1, num_classe=params.n_class)
    encoder_z = Encoder_Z(channel_in=1, latent_size=params.latent_size)
    decoder = Decoder(z_size=params.n_class + params.latent_size, channel_num=64)
    discriminator = Discriminator(channel_in=1, recon_level=params.recon_level)
    encoder_y.apply(weights_init)
    encoder_z.apply(weights_init)
    decoder.apply(weights_init)
    discriminator.apply(weights_init)
    encoder_y = encoder_y.to(device)
    encoder_z = encoder_z.to(device)
    decoder = decoder.to(device)
    discriminator = discriminator.to(device)

    # define the optimizers
    optim_encoder_y = optim.RMSprop(encoder_y.parameters(), lr=params.lr_encoder_y, alpha=0.9)
    optim_encoder_z = optim.RMSprop(encoder_z.parameters(), lr=params.lr_encoder_z, alpha=0.9)
    optim_decoder = optim.RMSprop(decoder.parameters(), lr=params.lr_decoder, alpha=0.9)
    optim_discriminator = optim.RMSprop(discriminator.parameters(), lr=params.lr_discriminator, alpha=0.9)

    # schedule learning rate
    Steps_y = [10000, 20000]
    lr_scheduler_encoder_y = MultiStepLR(optim_encoder_y, milestones=Steps_y, gamma=params.decay_lr)
    Steps = [10000, 20000, 30000, 50000, 70000, 100000]
    lr_scheduler_encoder_z = MultiStepLR(optim_encoder_z, milestones=Steps, gamma=params.decay_lr)
    lr_scheduler_decoder = MultiStepLR(optim_decoder, milestones=Steps, gamma=params.decay_lr)
    lr_scheduler_dis = MultiStepLR(optim_discriminator, milestones=Steps, gamma=params.decay_lr)

    # define several loss functions
    digit_class = nn.CrossEntropyLoss().to(device)  # classification
    # decorr_regul = XCov().to(device)  # decorrelation regularization
    decorr_regul = dCov2().to(device)

    # train the whole model
    count_update_step = training(params, encoder_y, encoder_z, decoder, discriminator,
                                 optim_encoder_y, optim_encoder_z, optim_decoder, optim_discriminator,
                                 lr_scheduler_encoder_y, lr_scheduler_encoder_z, lr_scheduler_decoder, lr_scheduler_dis,
                                 device, digit_class, decorr_regul, train_loader, valid_loader,
                                 margin, equilibrium, save_dir)
    visual_recons(encoder_y, encoder_z, decoder, device, test_loader)

    # save the settings
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write('Settings and Results:\n')
        f.write('------------------------------\n')
        f.write('Random seed = ' + str(params.seed) + '\n')
        f.write('Image size = ' + str(params.img_sz) + '\n')
        f.write('Number of classes = ' + str(params.n_class) + '\n')
        f.write('Latent vector size = ' + str(params.latent_size) + '\n')
        f.write('Feature reconstruction level = ' + str(params.recon_level) + '\n')
        f.write('Batch size = ' + str(params.batch_size) + '\n')
        f.write('------------------------------\n')
        f.write('Learning rate of Encoder_Y = ' + str(params.lr_encoder_y) + '\n')
        f.write('Learning rate of Encoder_Z = ' + str(params.lr_encoder_z) + '\n')
        f.write('Learning rate of Decoder = ' + str(params.lr_decoder) + '\n')
        f.write('Learning rate of Discriminator = ' + str(params.lr_discriminator) + '\n')
        f.write('Learning rate decay = ' + str(params.decay_lr) + '\n')
        f.write('------------------------------\n')
        f.write('Regularization parameter of the decorrelation term  = ' + str(params.lambda_decorr) + '\n')
        f.write('Progressively increase decorrelation lambda  = ' + str(params.lambda_schedule) + '\n')
        f.write('Regularization parameter of the reconstruction term in decoder  = ' + str(params.lambda_recons) + '\n')
        f.write('Regularization parameter of the discriminator term  = ' + str(params.lambda_dis) + '\n')
        f.write('------------------------------\n')
        f.write('Training samples = ' + str(params.n_train) + '\n')
        f.write('Validation samples = ' + str(params.n_valid) + '\n')
        f.write('Test samples = ' + str(params.n_test) + '\n')
        f.write('------------------------------\n')
        f.write('Max epoch for Encoder_Y = ' + str(params.n_epochs_EncY) + '\n')
        f.write('Max epoch = ' + str(params.n_epochs) + '\n')
        f.write('Total update steps = ' + str(count_update_step) + '\n')


if __name__ == '__main__':
    main()
