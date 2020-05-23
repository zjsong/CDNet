"""
AE-XCov for learning disentangled representations on MNIST dataset.
"""


import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from visdom import Visdom
from aexcov_model import Encoder, Decoder
from src.cross_covariance import XCov
from src.distance_covariance import dCov2
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


# model training function
def training(params, encoder, decoder, optim_encoder, optim_decoder, device,
             digit_class, decorr_regul, train_loader, valid_loader, save_dir):

    encoder.train()
    decoder.train()

    count_update_step = 0
    indices = torch.LongTensor(params.batch_size, 1)
    labels_onehot = torch.FloatTensor(params.batch_size, params.n_class)

    for i in range(params.n_epochs):

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):

            batch_x = batch_x.to(device)

            # convert the labels into one-hot form vectors
            indices.zero_()
            indices = batch_y.view(-1, 1)
            labels_onehot.zero_()
            labels_onehot.scatter_(1, indices, 1)
            batch_y_onehot = labels_onehot.to(device)
            batch_y = batch_y.to(device)

            count_update_step += 1

            ############################
            # (1) update encoder
            ############################
            for p in encoder.parameters():
                p.requires_grad_(True)
            for p in decoder.parameters():
                p.requires_grad_(False)
            optim_encoder.zero_grad()

            y_class, z_latent = encoder(batch_x)
            x_recons = decoder(batch_y_onehot, z_latent)

            # classification loss
            loss_class = digit_class(y_class, batch_y)
            # decorrelation loss
            y_class_softmax = F.softmax(y_class, 1)
            loss_decorr = decorr_regul(y_class_softmax, z_latent)
            # image reconstruction loss
            loss_recons_image = torch.mean(0.5 * (batch_x.view(len(batch_x), -1)
                                                  - x_recons.view(len(x_recons), -1)) ** 2)
            loss_encoder = params.lambda_class * loss_class + params.lambda_decorr * loss_decorr \
                           + params.lambda_recons * loss_recons_image

            loss_encoder.backward()
            optim_encoder.step()

            ############################
            # (2) update decoder
            ############################
            for p in decoder.parameters():
                p.requires_grad_(True)
            for p in encoder.parameters():
                p.requires_grad_(False)
            optim_decoder.zero_grad()

            _, z_latent = encoder(batch_x)
            x_recons = decoder(batch_y_onehot, z_latent)

            # image reconstruction loss
            loss_recons_image = torch.mean(0.5 * (batch_x.view(len(batch_x), -1)
                                                  - x_recons.view(len(x_recons), -1)) ** 2)
            loss_decoder = params.lambda_recons * loss_recons_image

            loss_decoder.backward()
            optim_decoder.step()

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

                y_value = np.asarray(loss_class.item()).reshape(1, )
                y_label = 'Loss'
                title = 'Cross Entropy Loss'
                legend = ['Loss_BCE']
                win_attrs_class = creat_vis_plot(viz, x_value, y_value,
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

                y_value = np.asarray(loss_class.item()).reshape(1, )
                update_vis(viz, win_attrs_class, x_value, y_value)

                y_value = np.asarray(loss_decorr.item()).reshape(1, )
                update_vis(viz, win_decorr, x_value, y_value)

            # evaluate the model
            if count_update_step % 1000 == 0:
                print('\nUpdate step: {:d}'
                      '\nmean loss_img_recons: {:.4f}'
                      '\nmean loss_class: {:.4f}'
                      '\nmean loss_decorr: {:.4f}'.format(
                    count_update_step, loss_recons_image.item(), loss_class.item(), loss_decorr.item()))

                # evaluation on validation set
                loss_recons, loss_decorr, class_error_rate = evaluate_classification(params, encoder, decoder,
                                                                                     valid_loader, params.n_valid,
                                                                                     decorr_regul, device)
                print('Classification Error Rate on Validation Set: {:.2f}%'.format(class_error_rate))
                print('Decorrelation Error on Each Mini-Batch: {:.4f}'.format(loss_decorr))
                print('Reconstruction Error on Each Sample: {:.4f}'.format(loss_recons))

                if count_update_step == 1000:
                    viz = Visdom()
                    x_value = np.asarray(count_update_step).reshape(1, )
                    x_label = 'Training Step'

                    y_value = np.asarray(loss_recons).reshape(1, )
                    y_label = 'Reconstruction Error'
                    title = 'Valid Image Reconstruction Error'
                    legend = ['Recons_Error']
                    win_img_recons_valid = creat_vis_plot(viz, x_value, y_value,
                                                          x_label, y_label, title, legend)

                    y_value = np.asarray(class_error_rate).reshape(1, )
                    y_label = 'Classification Error'
                    title = 'Valid Classification Error'
                    legend = ['Class_Error']
                    win_class_valid = creat_vis_plot(viz, x_value, y_value,
                                                     x_label, y_label, title, legend)

                    y_value = np.asarray(loss_decorr).reshape(1, )
                    y_label = 'Decorrelation Error'
                    title = 'Valid Decorrelation Error'
                    legend = ['Decorr_Error']
                    win_decorr_valid = creat_vis_plot(viz, x_value, y_value,
                                                      x_label, y_label, title, legend)
                else:
                    x_value = np.asarray(count_update_step).reshape(1, )

                    y_value = np.asarray(loss_recons).reshape(1, )
                    update_vis(viz, win_img_recons_valid, x_value, y_value)

                    y_value = np.asarray(class_error_rate).reshape(1, )
                    update_vis(viz, win_class_valid, x_value, y_value)

                    y_value = np.asarray(loss_decorr).reshape(1, )
                    update_vis(viz, win_decorr_valid, x_value, y_value)

                encoder.train()
                decoder.train()

            # save the midterm model sates
            if count_update_step % 5000 == 0:
                torch.save(encoder.state_dict(),
                           save_dir + '/encoder_step' + str(count_update_step) + '.pt')
                torch.save(decoder.state_dict(),
                           save_dir + '/decoder_step' + str(count_update_step) + '.pt')

            # lr_scheduler_encoder.step()
            # lr_scheduler_decoder.step()

    # save the whole model
    torch.save(encoder.state_dict(), save_dir + '/encoder_final.pt')
    torch.save(decoder.state_dict(), save_dir + '/decoder_final.pt')

    return count_update_step


# test function
def evaluate_classification(params, encoder, decoder, data_loader, num_data, decorr_regul, device):
    encoder.eval()
    decoder.eval()

    correct_class = 0
    loss_decorr = 0
    loss_recons = 0
    num_batch = 0

    with torch.no_grad():
        indices = torch.LongTensor(params.batch_size, 1)
        labels_onehot = torch.FloatTensor(params.batch_size, params.n_class)
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)

            # convert the labels into one-hot form vectors
            indices.zero_()
            indices = batch_y.view(-1, 1)
            labels_onehot.zero_()
            labels_onehot.scatter_(1, indices, 1)
            batch_y_onehot = labels_onehot.to(device)
            batch_y = batch_y.to(device)

            num_batch += 1

            # ---------------- classification error rate -----------------
            y_class, z_latent = encoder(batch_x)
            pred = y_class.max(1, keepdim=True)[1]
            correct_class += pred.eq(batch_y.view_as(pred)).cpu().sum().item()

            # ---------------- reconstruction error -----------------
            x_recons = decoder(batch_y_onehot, z_latent)
            loss_recons += F.mse_loss(x_recons, batch_x).item()

            # ---------------- decorrelation -----------------
            y_class_softmax = F.softmax(y_class, 1)
            loss_decorr += decorr_regul(y_class_softmax, z_latent).item()

    loss_recons /= num_data
    loss_decorr /= num_batch
    class_error_rate = 100 * (1 - correct_class / num_data)
    return loss_recons, loss_decorr, class_error_rate


# visualize the reconstruction results
def visual_recons(params, encoder, decoder, device, data_loader):
    encoder.eval()
    decoder.eval()

    n = 5
    plt.figure(figsize=(10, 4))

    with torch.no_grad():
        indices = torch.LongTensor(params.batch_size, 1)
        labels_onehot = torch.FloatTensor(params.batch_size, params.n_class)
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)

            # convert the labels into one-hot form vectors
            indices.zero_()
            indices = batch_y.view(-1, 1)
            labels_onehot.zero_()
            labels_onehot.scatter_(1, indices, 1)
            batch_y_onehot = labels_onehot.to(device)
            batch_y = batch_y.to(device)

            _, z_latent = encoder(batch_x)
            outputs = decoder(batch_y_onehot, z_latent)

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
            label_original = batch_y_onehot[subject, :].view(1, -1)
            # print(label_original)
            ax = plt.subplot(1, 3, 1)
            plt.imshow(image_test)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # reconstructed image
            output_test = decoder(label_original, z_test)
            output_test = output_test[0].cpu().numpy().reshape(28, 28)
            ax = plt.subplot(1, 3, 2)
            plt.imshow(output_test)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # manipulating image
            curr_digit_id = batch_y[subject].item()
            desir_digit_id = 5
            label_test = label_original.clone().cpu().numpy()
            curr_digit_value = label_test[0, curr_digit_id]
            desir_digit_value = label_test[0, desir_digit_id]
            label_test[0, curr_digit_id] = desir_digit_value
            label_test[0, desir_digit_id] = curr_digit_value
            label_test = torch.from_numpy(label_test).to(device)
            output_test = decoder(label_test, z_test)
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
    save_dir = 'results/aexcov'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # parse parameters
    parser = argparse.ArgumentParser(description='AE-XCov')
    parser.add_argument('--img_sz', type=int, default=28,
                        help='Image size (images have to be squared)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size')
    parser.add_argument('--n_class', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--latent_size', type=int, default=10,
                        help='Latent vector size')
    parser.add_argument('--n_epochs', type=int, default=150,
                        help='Total number of epochs')
    parser.add_argument('--n_train', type=int, default=50000,
                        help='The number of training samples')
    parser.add_argument('--n_valid', type=int, default=10000,
                        help='The number of validation samples')
    parser.add_argument('--n_test', type=int, default=10000,
                        help='The number of test samples')
    parser.add_argument('--lr_encoder', type=float, default=1e-4,
                        help='Learning rate for encoder')
    parser.add_argument('--lr_decoder', type=float, default=1e-4,
                        help='Learning rate for decoder')
    parser.add_argument('--lambda_class', type=float, default=1,
                        help='Image classification coefficient for encoder')
    parser.add_argument('--lambda_decorr', type=float, default=5,
                        help='Decorrelation regularization coefficient')
    parser.add_argument('--lambda_recons', type=float, default=1,
                        help='Feature reconstruction coefficient for decoder')
    parser.add_argument('--decay_lr', type=float, default=0.75,
                        help='Learning rate decay')
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
    encoder = Encoder(channel_in=1, num_classe=params.n_class, latent_size=params.latent_size)
    decoder = Decoder(z_size=params.n_class + params.latent_size, channel_num=64)
    encoder.apply(weights_init)
    decoder.apply(weights_init)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # define the optimizers
    optim_encoder = optim.RMSprop(encoder.parameters(), lr=params.lr_encoder, alpha=0.9)
    optim_decoder = optim.RMSprop(decoder.parameters(), lr=params.lr_decoder, alpha=0.9)

    # define several loss functions
    digit_class = nn.CrossEntropyLoss().to(device)  # classification
    decorr_regul = XCov().to(device)  # decorrelation regularization
    # decorr_regul = dCov2().to(device)

    # train the whole model
    count_update_step = training(params, encoder, decoder, optim_encoder, optim_decoder, device,
                                 digit_class, decorr_regul, train_loader, valid_loader, save_dir)

    visual_recons(params, encoder, decoder, device, test_loader)

    # save the settings
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write('Settings and Results:\n')
        f.write('------------------------------\n')
        f.write('Random seed = ' + str(params.seed) + '\n')
        f.write('Image size = ' + str(params.img_sz) + '\n')
        f.write('Attribute vector size = ' + str(params.n_class) + '\n')
        f.write('Latent vector size = ' + str(params.latent_size) + '\n')
        f.write('Batch size = ' + str(params.batch_size) + '\n')
        f.write('------------------------------\n')
        f.write('Learning rate of Encoder = ' + str(params.lr_encoder) + '\n')
        f.write('Learning rate of Decoder = ' + str(params.lr_decoder) + '\n')
        f.write('Learning rate decay = ' + str(params.decay_lr) + '\n')
        f.write('------------------------------\n')
        f.write('Regularization parameter of the classification term  = ' + str(params.lambda_class) + '\n')
        f.write('Regularization parameter of the decorrelation term  = ' + str(params.lambda_decorr) + '\n')
        f.write('Regularization parameter of the reconstruction term in decoder  = ' + str(params.lambda_recons) + '\n')
        f.write('------------------------------\n')
        f.write('Training samples = ' + str(params.n_train) + '\n')
        f.write('Validation samples = ' + str(params.n_valid) + '\n')
        f.write('Test samples = ' + str(params.n_test) + '\n')
        f.write('------------------------------\n')
        f.write('Max epoch = ' + str(params.n_epochs) + '\n')
        f.write('Total update steps = ' + str(count_update_step) + '\n')


if __name__ == '__main__':
    main()
