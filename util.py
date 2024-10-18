import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# transforms

## JPEG & Sharpening

train_transform = [transforms.ToTensor()]
test_transform = [transforms.ToTensor()]
train_transform = transforms.Compose(train_transform)
test_transform = transforms.Compose(test_transform)

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
        unlearnable_dataset = datasets.CIFAR10(root='.',
                                               train=True,
                                               download=True,
                                               transform=train_transform)
        batch_size = 500
    else:
        unlearnable_dataset = datasets.CIFAR100(root='.',
                                                train=True,
                                                download=True,
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
