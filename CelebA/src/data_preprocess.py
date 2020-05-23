"""
Read and preprocess CelebA images and attributes.
"""


import os
import numpy as np
import cv2
from PIL import Image
from skimage import filters, transform


# images path
original_imgs_root = 'D:/Projects/dataset/CelebA/data_imgs_attrs/img_align_celeba/'
save_imgs_root = 'D:/Projects/dataset/CelebA/data_imgs_attrs_64_64_clip/imgs/'

# attributes path
original_attrs_root = 'D:/Projects/dataset/CelebA/data_imgs_attrs/list_attr_celeba.txt'
save_attrs_root = 'D:/Projects/dataset/CelebA/data_imgs_attrs_64_64_clip/'

num_images = 202599
resize_size = 64
bbox = (40, 218 - 30, 15, 178 - 15)

if not os.path.exists(save_imgs_root):
    os.makedirs(save_imgs_root)
imgs_list = os.listdir(original_imgs_root)


def resize_images():

    for i in range(num_images):
        img = cv2.imread(original_imgs_root + imgs_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        # Smooth image before resize to avoid moire patterns
        scale = img.shape[0] / float(resize_size)
        sigma = np.sqrt(scale) / 2.0
        img = filters.gaussian(img, sigma=sigma, multichannel=True)
        img = transform.resize(img, (resize_size, resize_size, 3), order=3, mode='constant')
        img = (img * 255).astype(np.uint8)

        img = Image.fromarray(img)
        img.save(save_imgs_root + imgs_list[i])

        if (i % 10000) == 0:
            print('%d images complete' % i)


def save_attrs():

    save_name = save_attrs_root + 'attrs_all.csv'
    attrs_all = []

    f = open(original_attrs_root)
    line = f.readline()
    attrs_name = f.readline()
    attrs_name = attrs_name.split()
    attrs_name.insert(0, 'image_name')
    attrs_all.append(attrs_name)
    line = f.readline()
    while line:
        array = line.split()
        attrs_all.append(array)
        line = f.readline()

    f.close()
    np.savetxt(save_name, attrs_all, fmt='%5s', delimiter=',')


resize_images()
save_attrs()
