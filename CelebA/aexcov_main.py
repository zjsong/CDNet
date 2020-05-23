"""
AE-XCov for learning disentangled representations on CelebA dataset.
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
from visdom import Visdom
from aexcov_model import Encoder, Decoder
from src.cross_covariance import XCov
from src.distance_covariance import dCov2
from src.data_loader import load_data
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
def training(params, encoder, decoder, optim_encoder, optim_decoder,
             lr_scheduler_encoder, lr_scheduler_decoder, device,
             attrs_class, decorr_regul, train_loader, valid_loader, save_dir):

    encoder.train()
    decoder.train()

    count_update_step = 0

    for i in range(params.n_epochs):

        for batch_idx, sample_batched in enumerate(train_loader):
            data, label = sample_batched['image'], sample_batched['attributes']

            batch_x = data.to(device)
            batch_y = label.to(device)

            count_update_step += 1

            ############################
            # (1) update encoder
            ############################
            for p in encoder.parameters():
                p.requires_grad_(True)
            for p in decoder.parameters():
                p.requires_grad_(False)
            optim_encoder.zero_grad()

            y_attrs, z_latent = encoder(batch_x)
            x_recons = decoder(batch_y, z_latent)

            # classification loss
            loss_class = attrs_class(y_attrs, batch_y)
            # decorrelation loss
            y_attrs_sigmoid = torch.sigmoid(y_attrs)
            loss_decorr = decorr_regul(y_attrs_sigmoid, z_latent)
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

            y_attrs, z_latent = encoder(batch_x)
            x_recons = decoder(batch_y, z_latent.detach())

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
                title = 'Binary Cross Entropy Loss'
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
                      '\nmean loss_attrs_class: {:.4f}'
                      '\nmean loss_decorr: {:.4f}'.format(
                    count_update_step, loss_recons_image.item(), loss_class.item(), loss_decorr.item()))

                # evaluation on validation set
                _, er_total_valid, \
                loss_decorr_valid, loss_recons_valid = evaluate_classification(encoder, decoder,
                                                                               valid_loader, params.n_valid, decorr_regul, device)
                print('Attribute Classification Error Rate on Validation Set: {:.2f}%'.format(er_total_valid))
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

                    y_value = np.asarray(er_total_valid).reshape(1, )
                    y_label = 'Classification Error'
                    title = 'Valid Attr. Classification Error'
                    legend = ['Class_Error']
                    win_attrs_class_valid = creat_vis_plot(viz, x_value, y_value,
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

                    y_value = np.asarray(er_total_valid).reshape(1, )
                    update_vis(viz, win_attrs_class_valid, x_value, y_value)

                    y_value = np.asarray(loss_decorr_valid).reshape(1, )
                    update_vis(viz, win_decorr_valid, x_value, y_value)

                encoder.train()
                decoder.train()

            # save the midterm model sates
            if count_update_step % 5000 == 0:
                torch.save(encoder.state_dict(),
                           save_dir + '/encoder_step' + str(count_update_step) + '.pt')
                torch.save(decoder.state_dict(),
                           save_dir + '/decoder_step' + str(count_update_step) + '.pt')
            if (10000 < count_update_step < 20000) and (count_update_step % 1000 == 0):
                torch.save(encoder.state_dict(),
                           save_dir + '/encoder_step' + str(count_update_step) + '.pt')
                torch.save(decoder.state_dict(),
                           save_dir + '/decoder_step' + str(count_update_step) + '.pt')

            lr_scheduler_encoder.step()
            lr_scheduler_decoder.step()

    # save the whole model
    torch.save(encoder.state_dict(), save_dir + '/encoder_final.pt')
    torch.save(decoder.state_dict(), save_dir + '/decoder_final.pt')

    return count_update_step


# test function
def evaluate_classification(encoder, decoder, data_loader, num_data, decorr_regul, device):
    encoder.eval()
    decoder.eval()

    correct_each_attr = np.zeros([1, 40])
    loss_decorr = 0
    loss_recons = 0
    num_batch = 0

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(data_loader):

            data, label = sample_batched['image'], sample_batched['attributes']

            batch_x = data.to(device)
            batch_y = label.to(device)
            num_batch += 1

            y_attrs, z_latent = encoder(batch_x)
            x_recons = decoder(batch_y, z_latent)

            # attribute classification
            y_attrs_sigmoid = torch.sigmoid(y_attrs)
            pred_attrs = y_attrs_sigmoid >= 0.5
            compar_mat = (pred_attrs.cpu().numpy() == batch_y.cpu().numpy())
            # classification accuracy of each attribute across all subjects
            correct_each_attr += compar_mat.sum(0).reshape(1, -1)

            # decorrelation
            loss_decorr += decorr_regul(y_attrs_sigmoid, z_latent).item()

            # reconstruction
            loss_recons += torch.sum(torch.mean(0.5 * (batch_x.view(len(batch_x), -1)
                                                       - x_recons.view(len(x_recons), -1)) ** 2, 1)).item()

    error_rate_each_attr = 100 * (1 - correct_each_attr / num_data)
    error_rate_total = np.mean(error_rate_each_attr)
    loss_decorr /= num_batch
    loss_recons /= num_data

    return error_rate_each_attr, error_rate_total, loss_decorr, loss_recons


# visualize the reconstruction results
def visual_recons(encoder, decoder, device, test_loader):
    encoder.eval()
    decoder.eval()

    n = 5
    plt.figure(figsize=(10, 4))

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(test_loader):

            data, label = sample_batched['image'], sample_batched['attributes']

            batch_x = data.to(device)
            batch_y = label.to(device)

            y_attrs, z_latent = encoder(batch_x)
            outputs = decoder(batch_y, z_latent)

            in_out = zip(batch_x, outputs)
            count = 0
            for image, output in in_out:
                count += 1
                if count > n:
                    break

                # display original images
                ax = plt.subplot(2, n, count)
                image = (image.cpu().numpy().transpose(1, 2, 0) + 1) / 2
                plt.imshow(image)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # display reconstruction
                ax = plt.subplot(2, n, n + count)
                image_recons = (output.cpu().numpy().transpose(1, 2, 0) + 1) / 2
                plt.imshow(image_recons)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            plt.show(block=False)

            # fix latent representation, and change attribute label
            # original image
            subject = 0
            image_test = (batch_x[subject].cpu().numpy().transpose(1, 2, 0) + 1) / 2
            z_test = z_latent[subject].view(1, -1)
            label_original = batch_y[subject, :].view(1, -1)
            print(label_original)
            plt.figure()
            plt.imshow(image_test)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.show(block=False)

            # reconstructed image
            output_test = decoder(label_original, z_test)
            output_test = (output_test[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2
            plt.figure()
            plt.imshow(output_test)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.show(block=False)

            # manipulating image
            label_test = label_original.clone()
            label_test[:, 15] = 1  # Eyeglasses
            print(label_test)
            output_test = decoder(label_test, z_test)
            output_test = (output_test[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2
            plt.figure()
            plt.imshow(output_test)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.show(block=False)

            encoder.train()
            decoder.train()
            break


def main():
    # saving path
    save_dir = 'results/aexcov'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # parse parameters
    parser = argparse.ArgumentParser(description='AE-XCov')
    parser.add_argument('--img_sz', type=int, default=64,
                        help='Image size (images have to be squared)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument("--attr_size", type=int, default=40,
                        help='Attribute vector size')
    parser.add_argument("--latent_size", type=int, default=1000,
                        help='Latent vector size')
    parser.add_argument('--n_epochs', type=int, default=25,
                        help='Total number of epochs')
    parser.add_argument('--n_train', type=int, default=162770,
                        help='The number of training samples')
    parser.add_argument('--n_valid', type=int, default=19867,
                        help='The number of validation samples')
    parser.add_argument('--n_test', type=int, default=19962,
                        help='The number of test samples')
    parser.add_argument('--lr_encoder', type=float, default=1e-4,
                        help='Learning rate for encoder')
    parser.add_argument('--lr_decoder', type=float, default=1e-4,
                        help='Learning rate for decoder')
    parser.add_argument('--lambda_class', type=float, default=1,
                        help='Image classification coefficient for encoder')
    parser.add_argument('--lambda_decorr', type=float, default=1,
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

    # split data
    train_index = params.n_train
    valid_index = params.n_train + params.n_valid
    test_index = params.n_train + params.n_valid + params.n_test
    indices_all_imgs = np.arange(test_index)
    train_indices = indices_all_imgs[:train_index]
    valid_indices = indices_all_imgs[train_index:valid_index]
    test_indices = indices_all_imgs[valid_index:test_index]
    attrs_dir = 'D:/Projects/dataset/CelebA/data_imgs_attrs_64_64_clip/attrs_all.csv'
    resized_imgs_dir = 'D:/Projects/dataset/CelebA/data_imgs_attrs_64_64_clip/imgs/'
    train_loader, valid_loader, test_loader = load_data(attrs_dir, resized_imgs_dir, params.batch_size,
                                                        train_indices, valid_indices, test_indices, use_cuda)

    # build the model
    encoder = Encoder(channel_in=3, attr_size=params.attr_size, latent_size=params.latent_size)
    decoder = Decoder(z_size=params.attr_size + params.latent_size, size=256)
    encoder.apply(weights_init)
    decoder.apply(weights_init)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # define the optimizers
    optim_encoder = optim.RMSprop(encoder.parameters(), lr=params.lr_encoder, alpha=0.9)
    optim_decoder = optim.RMSprop(decoder.parameters(), lr=params.lr_decoder, alpha=0.9)

    # schedule learning rate
    Steps = [10000, 20000]
    lr_scheduler_encoder = MultiStepLR(optim_encoder, milestones=Steps, gamma=params.decay_lr)
    lr_scheduler_decoder = MultiStepLR(optim_decoder, milestones=Steps, gamma=params.decay_lr)

    # define several loss functions
    attrs_class = nn.BCEWithLogitsLoss().to(device)  # attribute classification
    decorr_regul = XCov().to(device)  # decorrelation regularization
    # decorr_regul = dCov2().to(device)

    # train the whole model
    count_update_step = training(params, encoder, decoder, optim_encoder, optim_decoder,
                                 lr_scheduler_encoder, lr_scheduler_decoder, device,
                                 attrs_class, decorr_regul, train_loader, valid_loader, save_dir)

    visual_recons(encoder, decoder, device, test_loader)

    # save the settings and results
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write('Settings and Results:\n')
        f.write('------------------------------\n')
        f.write('Random seed = ' + str(params.seed) + '\n')
        f.write('Image size = ' + str(params.img_sz) + '\n')
        f.write('Attribute vector size = ' + str(params.attr_size) + '\n')
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
