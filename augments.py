import torch
import numpy as np
from torchvision import transforms
from torch_dct import dct as torch_dct_dct, idct as torch_dct_idct


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


def get_dct_image(image, percentage, norm='ortho'):
    """
    image: torch tensor
    percentage: percentage of rows and columns to mask
    Returns:
    masked dct transformed image tensor of shape (3, H, W)
    """
    dct_rows = torch_dct_dct(image, norm=norm)
    dct_cols = torch_dct_dct(dct_rows.transpose(1, 2),
                             norm=norm).transpose(1, 2)

    C, H, W = dct_cols.shape
    num_rows = int(H * percentage / 100)
    num_cols = int(W * percentage / 100)
    mask = np.zeros_like(dct_cols)
    mask[:, :num_rows, :num_cols] = 1
    mask = dct_cols * mask

    idct_rows = torch_dct_idct(mask, norm=norm)
    idct_cols = torch_dct_idct(idct_rows.transpose(1, 2),
                               norm=norm).transpose(1, 2)
    idct_cols = torch.clamp(idct_cols, 0, 1)

    return idct_cols


# The default ff_percentage value is 30% based on the results of our paper. Feel free to change and experiment!
def get_transform(transform,
                  dataset,
                  normalize,
                  sharp_center,
                  ff_percentage=30):

    LambdaRSK = [
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: sharpen_image(x, center_mean=sharp_center, random=True))
    ]
    LambdaSSK = [
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: sharpen_image(x, center_mean=sharp_center, random=False))
    ]
    LambdaFF = [transforms.Lambda(lambda x: get_dct_image(x, ff_percentage))]

    augs = ff_augs = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        augs = augs + [transforms.ToTensor()]
    else:
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        augs = augs + [transforms.RandomRotation(15), transforms.ToTensor()]
        ff_augs += [transforms.RandomRotation(15)]

    base = [transforms.ToTensor()]
    if transform == 'ssk':
        train_transform = LambdaSSK + augs
    elif transform == 'ssk_ff':
        train_transform = LambdaSSK + LambdaFF + ff_augs
    elif transform == 'rsk':
        train_transform = LambdaRSK + augs
    elif transform == 'rsk_ff':
        train_transform = LambdaRSK + LambdaFF + ff_augs
    elif transform == 'ff':
        train_transform = base + LambdaFF + ff_augs
    else:
        train_transform = base  # clean

    if normalize:
        norm = [transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(train_transform + norm)
        test_transform = transforms.Compose(base + norm)
    else:
        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(base)

    return train_transform, test_transform
