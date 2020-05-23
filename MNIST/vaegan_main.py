"""
Learning VAE/GAN model on MNIST dataset.
"""


import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from vaegan_model import VaeGan
from visdom import Visdom
from src.utils import creat_vis_plot, update_vis


# visualize the reconstruction results
def visual_recons(net, device, test_loader):
    net.eval()

    n = 10
    plt.figure(figsize=(10, 2))

    with torch.no_grad():
        for j, (data_batch, target_batch) in enumerate(test_loader):

            data = data_batch.to(device)

            data_recons, _ = net(data)

            in_out = zip(data, data_recons)
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

            net.train()

            break


# test function
def evaluate_learning(net, data_loader, num_data, device):
    net.eval()

    loss_recons = 0
    with torch.no_grad():
        for j, (data_batch, target_batch) in enumerate(data_loader):

            batch_x = data_batch.to(device)

            x_recons, _ = net(batch_x)

            # reconstruction
            loss_recons += torch.sum(torch.mean(0.5 * (batch_x.view(len(batch_x), -1)
                                                       - x_recons.view(len(x_recons), -1)) ** 2, 1)).item()
    loss_recons /= num_data

    return loss_recons


if __name__ == "__main__":

    # saving path
    save_dir = 'results/vaegan'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    parser = argparse.ArgumentParser(description="VAE/GAN")
    parser.add_argument("--img_sz", type=int, default=28,
                        help="Image size (images have to be squared)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size")
    parser.add_argument("--z_size", type=int, default=20,
                        help="Latent vector size")
    parser.add_argument("--recon_level", type=int, default=2,
                        help="Level wherein computing feature reconstruction error")
    parser.add_argument("--n_epochs", type=int, default=250,
                        help="Total number of epochs")
    parser.add_argument('--n_train', type=int, default=50000,
                        help='The number of training samples')
    parser.add_argument('--n_valid', type=int, default=10000,
                        help='The number of validation samples')
    parser.add_argument('--n_test', type=int, default=10000,
                        help='The number of test samples')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--decay_lr", type=float, default=1,
                        help="Learning rate decay")
    parser.add_argument("--lambda_mse", type=float, default=1e-2,
                        help="Feature reconstruction coefficient for decoder")
    parser.add_argument('--seed', type=int, default=8,
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
    net = VaeGan(device=device, z_size=params.z_size, recon_level=params.recon_level).to(device)

    # define the optimizers
    optim_encoder = optim.RMSprop(params=net.encoder.parameters(), lr=params.lr, alpha=0.9)
    optim_decoder = optim.RMSprop(params=net.decoder.parameters(), lr=params.lr, alpha=0.9)
    optim_discriminator = optim.RMSprop(params=net.discriminator.parameters(), lr=params.lr, alpha=0.9)

    # train the whole model
    net.train()
    count_update_step = 0

    for i in range(params.n_epochs):

        for j, (data_batch, target_batch) in enumerate(train_loader):

            # target and input are the same images
            data = data_batch.to(device)

            data_recons, out_labels, out_layer, mus, variances = net(data)

            # split so we can get the different parts
            out_layer_predicted = out_layer[:len(out_layer) // 2]
            out_layer_original = out_layer[len(out_layer) // 2:]
            out_labels_original = out_labels[:len(out_labels) // 2]
            out_labels_sampled = out_labels[len(out_labels) // 2:]

            # losses
            nle_value, kl_value, mse_value, \
            dis_original, dis_sampled = net.loss(data, data_recons,
                                                 out_layer_original, out_layer_predicted,
                                                 out_labels_original, out_labels_sampled,
                                                 mus, variances)

            loss_encoder = torch.sum(kl_value) + torch.sum(mse_value)
            loss_discriminator = torch.sum(dis_original) + torch.sum(dis_sampled)
            loss_decoder = torch.sum(params.lambda_mse * mse_value) - (1.0 - params.lambda_mse) * loss_discriminator

            # selectively disable the decoder of the discriminator if they are unbalanced
            train_dis = True
            train_dec = True
            if torch.mean(dis_original).item() < equilibrium - margin \
                    or torch.mean(dis_sampled).item() < equilibrium - margin:
                train_dis = False
            if torch.mean(dis_original).item() > equilibrium + margin \
                    or torch.mean(dis_sampled).item() > equilibrium + margin:
                train_dec = False
            if train_dec is False and train_dis is False:
                train_dis = True
                train_dec = True

            # BACKPROP
            net.zero_grad()

            # encoder
            loss_encoder.backward(retain_graph=True)
            optim_encoder.step()

            net.zero_grad()

            # decoder
            if train_dec:
                loss_decoder.backward(retain_graph=True)
                optim_decoder.step()
                net.discriminator.zero_grad()

            # discriminator
            if train_dis:
                loss_discriminator.backward()
                optim_discriminator.step()

            # visualize losses
            count_update_step += 1
            if count_update_step == 1:
                viz = Visdom()
                x_value = np.asarray(count_update_step).reshape(1, )
                x_label = 'Training Step'

                y_value = np.asarray(torch.mean(nle_value).item()).reshape(1, )
                y_label = 'MSE'
                title = 'Image Reconstruction Loss'
                legend = ['MSE']
                win_img_recons = creat_vis_plot(viz, x_value, y_value,
                                                x_label, y_label, title, legend)

                y_value = np.asarray(torch.mean(mse_value).item()).reshape(1, )
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

            elif count_update_step % 50 == 0:
                x_value = np.asarray(count_update_step).reshape(1, )

                y_value = np.asarray(torch.mean(nle_value).item()).reshape(1, )
                update_vis(viz, win_img_recons, x_value, y_value)

                y_value = np.asarray(torch.mean(mse_value).item()).reshape(1, )
                update_vis(viz, win_feature_recons, x_value, y_value)

                y_value = np.column_stack((np.asarray(loss_discriminator.item()), np.asarray(loss_decoder.item())))
                update_vis(viz, win_dis_gen, x_value, y_value)

            # evaluate the model
            if count_update_step % 1000 == 0:
                print('\nUpdate step: {:d}'
                      '\nmean loss_img_recons: {:.4f}'
                      '\nmean loss_feature_recons: {:.4f}'
                      '\nmean loss_GAN_dis: {:.4f}'
                      '\nmean loss_decoder: {:.4f}'.format(
                    count_update_step, torch.mean(nle_value).item(), torch.mean(mse_value).item(),
                    loss_discriminator.item(), loss_decoder.item()))

                # evaluation on validation set
                loss_recons_valid = evaluate_learning(net, valid_loader, params.n_valid, device)
                print('Reconstruction Error on Each Valid Sample: {:.4f}'.format(loss_recons_valid))

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
                else:
                    x_value = np.asarray(count_update_step).reshape(1, )

                    y_value = np.asarray(loss_recons_valid).reshape(1, )
                    update_vis(viz, win_img_recons_valid, x_value, y_value)

                net.train()

            # save the midterm model sates
            if count_update_step % 5000 == 0:
                torch.save(net.state_dict(), save_dir + '/VAEGAN_step' + str(count_update_step) + '.pt')
                # visual_recons(net, device, test_loader)

    visual_recons(net, device, test_loader)

    # save the whole model
    torch.save(net.state_dict(), save_dir + '/VAEGAN_final.pt')

    # save the settings and results
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write('Settings and Results:\n')
        f.write('------------------------------\n')
        f.write('Random seed = ' + str(params.seed) + '\n')
        f.write('Image size = ' + str(params.img_sz) + '\n')
        f.write('Latent vector size = ' + str(params.z_size) + '\n')
        f.write('Feature reconstruction level = ' + str(params.recon_level) + '\n')
        f.write('Batch size = ' + str(params.batch_size) + '\n')
        f.write('------------------------------\n')
        f.write('Learning rate = ' + str(params.lr) + '\n')
        f.write('Learning rate decay = ' + str(params.decay_lr) + '\n')
        f.write('Regularization parameter of the reconstruction term in decoder  = ' + str(params.lambda_mse) + '\n')
        f.write('------------------------------\n')
        f.write('Training samples = ' + str(params.n_train) + '\n')
        f.write('Validation samples = ' + str(params.n_valid) + '\n')
        f.write('Test samples = ' + str(params.n_test) + '\n')
        f.write('------------------------------\n')
        f.write('Max epoch = ' + str(params.n_epochs) + '\n')
        f.write('Total update steps = ' + str(count_update_step) + '\n')
