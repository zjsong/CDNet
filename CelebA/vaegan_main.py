"""
Learning VAE/GAN model on CelebA dataset.
"""


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from src.data_loader import load_data
from vaegan_model import VaeGan
from visdom import Visdom
from src.utils import creat_vis_plot, update_vis


# visualize the reconstruction results
def visual_recons(net, device, test_loader):
    net.eval()

    n = 10
    plt.figure(figsize=(10, 2))

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(test_loader):

            data = sample_batched['image']
            data = data.to(device)

            data_recons, _ = net(data)

            in_out = zip(data, data_recons)
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

            net.train()

            break


# test function
def evaluate_learning(net, data_loader, num_data, device):
    net.eval()

    loss_recons = 0
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(data_loader):
            data = sample_batched['image']
            batch_x = data.to(device)

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
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--n_epochs", default=15, action="store", type=int, dest="n_epochs")
    parser.add_argument("--z_size", default=128, action="store", type=int, dest="z_size")
    parser.add_argument("--recon_level", default=3, action="store", type=int, dest="recon_level")
    parser.add_argument('--n_train', type=int, default=162770,
                        help='The number of training samples')
    parser.add_argument('--n_valid', type=int, default=19867,
                        help='The number of validation samples')
    parser.add_argument('--n_test', type=int, default=19962,
                        help='The number of test samples')
    parser.add_argument("--lambda_mse", default=2e-6, action="store", type=float, dest="lambda_mse")
    parser.add_argument("--lambda_dis", default=5e-1, action="store", type=float, dest="lambda_dis")
    parser.add_argument("--lr", default=3e-4, action="store", type=float, dest="lr")
    parser.add_argument("--decay_lr", default=0.75, action="store", type=float, dest="decay_lr")
    parser.add_argument("--decay_mse", default=1, action="store", type=float, dest="decay_mse")
    parser.add_argument("--decay_margin", default=1, action="store", type=float, dest="decay_margin")
    parser.add_argument("--decay_equilibrium", default=1, action="store", type=float, dest="decay_equilibrium")
    parser.add_argument('--seed', type=int, default=8, help='Random seed (default: 1)')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    batch_size = args.batch_size
    z_size = args.z_size
    recon_level = args.recon_level
    n_train = args.n_train
    n_valid = args.n_valid
    n_test = args.n_test
    decay_mse = args.decay_mse
    decay_margin = args.decay_margin
    n_epochs = args.n_epochs
    lambda_mse = args.lambda_mse
    lambda_dis = args.lambda_dis
    lr = args.lr
    decay_lr = args.decay_lr
    decay_equilibrium = args.decay_equilibrium

    # use GPUs
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    net = VaeGan(device=device, z_size=z_size, recon_level=recon_level).to(device)

    # DATASET
    train_index = n_train
    valid_index = n_train + n_valid
    test_index = n_train + n_valid + n_test
    indices_all_imgs = np.arange(test_index)
    train_indices = indices_all_imgs[:train_index]
    valid_indices = indices_all_imgs[train_index:valid_index]
    test_indices = indices_all_imgs[valid_index:test_index]
    attrs_dir = 'D:/Projects/dataset/CelebA/data_imgs_attrs_64_64_clip/attrs_all.csv'
    resized_imgs_dir = 'D:/Projects/dataset/CelebA/data_imgs_attrs_64_64_clip/imgs/'
    dataloader, dataloader_valid, dataloader_test = load_data(attrs_dir, resized_imgs_dir, batch_size,
                                                              train_indices, valid_indices, test_indices, args)

    # margin and equilibirum
    margin = 0.35
    equilibrium = 0.68

    # OPTIM-LOSS
    optimizer_encoder = optim.RMSprop(params=net.encoder.parameters(), lr=lr,
                                      alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    optimizer_decoder = optim.RMSprop(params=net.decoder.parameters(), lr=lr,
                                      alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    optimizer_discriminator = optim.RMSprop(params=net.discriminator.parameters(), lr=lr,
                                            alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)

    Steps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    lr_encoder = MultiStepLR(optimizer_encoder, milestones=Steps, gamma=decay_lr)
    lr_decoder = MultiStepLR(optimizer_decoder, milestones=Steps, gamma=decay_lr)
    lr_discriminator = MultiStepLR(optimizer_discriminator, milestones=Steps, gamma=decay_lr)

    count_update_step = 0
    for i in range(n_epochs):

        for j, sample_batched in enumerate(dataloader):

            net.train()

            # target and input are the same images
            data = sample_batched['image']
            batch_x = data.cuda()

            # get output
            out, out_labels, out_layer, mus, variances = net(batch_x)
            out_layer_recons = out_layer[:len(out_layer) // 2]
            out_layer_original = out_layer[len(out_layer) // 2:]
            out_labels_sampled = out_labels[:len(out_labels) // 2]
            out_labels_original = out_labels[len(out_labels) // 2:]

            # losses
            nle_value, kl_value, mse_value, bce_dis_original_value, \
            bce_dis_sampled_value, bce_gen_sampled = net.loss(batch_x, out,
                                                              out_layer_original, out_layer_recons,
                                                              out_labels_original, out_labels_sampled,
                                                              mus, variances)

            loss_encoder = torch.sum(kl_value) + torch.sum(mse_value)
            loss_discriminator = torch.sum(bce_dis_original_value) + torch.sum(bce_dis_sampled_value)
            loss_decoder = torch.sum(lambda_mse * mse_value) - lambda_dis * loss_discriminator

            # selectively disable the decoder of the discriminator if they are unbalanced
            train_dis = True
            train_dec = True
            if torch.mean(bce_dis_original_value).item() < equilibrium-margin \
                    or torch.mean(bce_dis_sampled_value).item() < equilibrium-margin:
                train_dis = False
            if torch.mean(bce_dis_original_value).item() > equilibrium+margin \
                    or torch.mean(bce_dis_sampled_value).item() > equilibrium+margin:
                train_dec = False
            if train_dec is False and train_dis is False:
                train_dis = True
                train_dec = True

            # BACKPROP
            net.zero_grad()
            # encoder
            loss_encoder.backward(retain_graph=True)
            optimizer_encoder.step()
            net.zero_grad()
            # decoder
            if train_dec:
                loss_decoder.backward(retain_graph=True)
                optimizer_decoder.step()
                net.discriminator.zero_grad()
            # discriminator
            if train_dis:
                loss_discriminator.backward()
                optimizer_discriminator.step()

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

                y_value = np.asarray(torch.sum(mse_value).item()).reshape(1, )
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

                y_value = np.asarray(torch.sum(mse_value).item()).reshape(1, )
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
                    count_update_step, torch.mean(nle_value).item(), torch.sum(mse_value).item(),
                    loss_discriminator.item(), loss_decoder.item()))

                # evaluation on validation set
                loss_recons_valid = evaluate_learning(net, dataloader_valid, n_valid, device)
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
            if (count_update_step >= 5000) and count_update_step % 1000 == 0:
                torch.save(net.state_dict(), save_dir + '/VAEGAN_step' + str(count_update_step) + '.pt')
                # visual_recons(net, device, dataloader_valid)

            lr_encoder.step()
            lr_decoder.step()
            lr_discriminator.step()
        margin *= decay_margin
        equilibrium *= decay_equilibrium
        if margin > equilibrium:
            equilibrium = margin
        lambda_mse *= decay_mse
        if lambda_mse > 1:
            lambda_mse = 1

    visual_recons(net, device, dataloader_test)

    # save the whole model
    torch.save(net.state_dict(), save_dir + '/VAEGAN_final.pt')

    # save the settings and results
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write('Settings and Results:\n')
        f.write('------------------------------\n')
        f.write('Latent vector size = ' + str(args.z_size) + '\n')
        f.write('Feature reconstruction level = ' + str(args.recon_level) + '\n')
        f.write('Batch size = ' + str(args.batch_size) + '\n')
        f.write('------------------------------\n')
        f.write('Learning rate = ' + str(args.lr) + '\n')
        f.write('Learning rate decay = ' + str(args.decay_lr) + '\n')
        f.write('Regularization parameter of the reconstruction term in decoder  = ' + str(args.lambda_mse) + '\n')
        f.write('------------------------------\n')
        f.write('Training samples = ' + str(args.n_train) + '\n')
        f.write('Validation samples = ' + str(args.n_valid) + '\n')
        f.write('Test samples = ' + str(args.n_test) + '\n')
        f.write('------------------------------\n')
        f.write('Max epoch = ' + str(args.n_epochs) + '\n')
        f.write('Total update steps = ' + str(count_update_step) + '\n')
