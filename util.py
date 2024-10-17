import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from io import BytesIO
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_dct import dct as torch_dct_dct, idct as torch_dct_idct
from typing import Dict, List, Tuple
from torchvision.datasets import ImageFolder

cifar_path = '/fs/vulcan-datasets/CIFAR'
imagenet_train_path = '/fs/vulcan-datasets/imagenet/train/'

# transforms

## JPEG & Sharpening

train_transform = [transforms.ToTensor()]
test_transform = [transforms.ToTensor()]
train_transform = transforms.Compose(train_transform)
test_transform = transforms.Compose(test_transform)


def JPEGcompression(image, jpeg):
    """
    jpeg: controls jpeg compression quality (100 for no compression, 10 for high compression)
    """
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=jpeg, optimize=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)


def sharpen_image(im, center_mean, random):
    """
    Given an image tensor sharpen it using a sharpening kernel
    im: image tensor of shape (3, H, W)
    center: the central value in the kernel
    Returns:
    sharpened image tensor of shape (3, H', W')
    """
    if random:
        stddev = 0.1
        center = np.random.normal(center_mean, stddev)
        neighbor_mean = (center - 1) / 4
        neighbor1 = np.random.normal(neighbor_mean, stddev)
        neighbor2 = np.random.normal(neighbor_mean, stddev)
        neighbor3 = np.random.normal(neighbor_mean, stddev)
        neighbor4 = np.random.normal(neighbor_mean, stddev)
        sharpen_kernel = torch.tensor(
            [[0, -neighbor1, 0], [-neighbor2, center, -neighbor3],
             [0, -neighbor4, 0]],
            dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    else:
        center = center_mean
        neighbor = (center - 1) / 4
        sharpen_kernel = torch.tensor(
            [[0, -neighbor, 0], [-neighbor, center, -neighbor],
             [0, -neighbor, 0]],
            dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    sharpen_kernel = sharpen_kernel.to(im.device)
    sharpened_im = torch.nn.functional.conv2d(im, sharpen_kernel,
                                              groups=3).clamp(0, 1)
    return sharpened_im


## DCT


def mask_dct(dct_image, percentage, imagenet=False):
    """
    dct_image: dct transformed image tensor of shape (3, H, W)
    percentage: percentage of rows and columns to mask
    Returns:
    masked dct transformed image tensor of shape (3, H, W)
    """

    C, H, W = dct_image.shape
    if imagenet:
        num_rows = int(H * .4 * percentage / 100)
        num_cols = int(W * .4 * percentage / 100)
    else:
        num_rows = int(H * percentage / 100)
        num_cols = int(W * percentage / 100)
    mask = np.zeros_like(dct_image)
    mask[:, :num_rows, :num_cols] = 1

    return dct_image * mask


def get_dct_image(image, percentage, imagenet=False, norm='ortho'):
    """
    image: torch tensor
    percentage: percentage of rows and columns to mask
    Returns:
    masked dct transformed image tensor of shape (3, H, W)
    """
    dct_rows = torch_dct_dct(image, norm=norm)
    dct_cols = torch_dct_dct(dct_rows.transpose(1, 2),
                             norm=norm).transpose(1, 2)

    mask = mask_dct(dct_cols, percentage, imagenet)

    idct_rows = torch_dct_idct(mask, norm=norm)
    idct_cols = torch_dct_idct(idct_rows.transpose(1, 2),
                               norm=norm).transpose(1, 2)
    idct_cols = torch.clamp(idct_cols, 0, 1)

    return idct_cols


# CUDA


def get_filter_unlearnable(blur_parameter,
                           seed,
                           num_cls,
                           mix,
                           dataset,
                           center_parameter=1.0,
                           grayscale=False,
                           kernel_size=3,
                           same=False):

    np.random.seed(seed)
    cnns = []
    with torch.no_grad():
        for i in range(num_cls):
            cnns.append(
                torch.nn.Conv2d(3, 3, kernel_size, groups=3, padding=1).cuda())
            if blur_parameter is None:
                blur_parameter = 1

            w = np.random.uniform(low=0,
                                  high=blur_parameter,
                                  size=(3, 1, kernel_size, kernel_size))
            if center_parameter is not None:
                shape = w[0][0].shape
                w[0, 0,
                  np.random.randint(shape[0]),
                  np.random.randint(shape[1])] = 1.0

            w[1] = w[0]
            w[2] = w[0]
            cnns[i].weight.copy_(torch.tensor(w))
            cnns[i].bias.copy_(cnns[i].bias * 0)

    cnns = np.stack(cnns)

    if same:
        cnns = np.stack([cnns[0]] * len(cnns))

    if dataset == 'cifar10':
        unlearnable_dataset = datasets.CIFAR10(root=cifar_path,
                                               train=True,
                                               download=False,
                                               transform=train_transform)
        batch_size = 500
    else:
        unlearnable_dataset = datasets.CIFAR100(root=cifar_path,
                                                train=True,
                                                download=False,
                                                transform=train_transform)
        batch_size = 500

    unlearnable_loader = DataLoader(dataset=unlearnable_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    drop_last=False,
                                    num_workers=4)

    pbar = tqdm(unlearnable_loader, total=len(unlearnable_loader))
    images_ = []

    for images, labels in pbar:
        images, labels = images.cuda(), labels.cuda()
        for i in range(len(images)):

            prob = np.random.random()
            if prob < mix:  # mix*100% of data is poisoned
                id = labels[i].item()
                img = cnns[id](images[i:i +
                                      1]).detach().cpu()  # convolve class-wise

                # # black and white
                if grayscale:
                    img_bw = img[0].mean(0)
                    img[0][0] = img_bw
                    img[0][1] = img_bw
                    img[0][2] = img_bw

                images_.append(img / img.max())
            else:
                images_.append(images[i:i + 1].detach().cpu())

    # making unlearnable data
    unlearnable_dataset.data = unlearnable_dataset.data.astype(np.float32)
    for i in range(len(unlearnable_dataset)):
        unlearnable_dataset.data[i] = images_[i][0].numpy().transpose(
            (1, 2, 0)) * 255
        unlearnable_dataset.data[i] = np.clip(unlearnable_dataset.data[i],
                                              a_min=0,
                                              a_max=255)
    unlearnable_dataset.data = unlearnable_dataset.data.astype(np.uint8)

    return unlearnable_dataset, cnns


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
