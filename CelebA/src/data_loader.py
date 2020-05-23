"""
Load and preprocess CelebA data.
"""


import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from .construct_celeba_dataset import CelebADataset


def load_data(attrs_dir, resized_imgs_dir, batch_size,
              train_indices, valid_indices, test_indices, use_cuda):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset_imgs = CelebADataset(attrs_dir, resized_imgs_dir, transform)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(dataset_imgs, batch_size=batch_size, shuffle=False,
                                               sampler=SubsetRandomSampler(train_indices), **kwargs)

    valid_loader = torch.utils.data.DataLoader(dataset_imgs, batch_size=batch_size, shuffle=False,
                                               sampler=SubsetRandomSampler(valid_indices), **kwargs)

    test_loader = torch.utils.data.DataLoader(dataset_imgs, batch_size=batch_size, shuffle=False,
                                              sampler=SubsetRandomSampler(test_indices), **kwargs)

    return train_loader, valid_loader, test_loader
