"""
Construct the CelebA dataset class.
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class CelebADataset(Dataset):

    def __init__(self, csv_file, imgs_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            imgs_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.attributes_frame = pd.read_csv(csv_file)
        self.images_dir = imgs_dir
        self.transform = transform

    def __len__(self):
        return len(self.attributes_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.attributes_frame.iloc[idx, 0])
        image = plt.imread(img_name)

        if self.transform:
            image = self.transform(image)

        attributes = self.attributes_frame.iloc[idx, 1:]

        # for 1 * 40 attribute label
        # convert the attribute values (1 and -1) into binary representations (1 and 0)
        attrs_binary = np.zeros(len(attributes))
        for id_attr, value_attr in enumerate(attributes):
            if value_attr == 1:
                attrs_binary[id_attr] = 1
            else:
                pass

        attrs_binary = torch.from_numpy(attrs_binary).type(torch.FloatTensor)

        sample = {'image': image, 'attributes': attrs_binary}
        return sample
