from torchvision import transforms
from util import sharpen_image, get_dct_image


# The default ff_percentage value is 30 based on the results of our paper. Feel free to change and experiment!
def get_transform(transform,
                  dataset,
                  normalize,
                  sharp_center,
                  ff_percentage=30):

    rsk = [
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: sharpen_image(x, center_mean=sharp_center, random=True))
    ]
    ssk = [
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: sharpen_image(x, center_mean=sharp_center, random=False))
    ]
    ff = [transforms.Lambda(lambda x: get_dct_image(x, ff_percentage))]

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
        train_transform = base + [ssk, augs]
    elif transform == 'ssk_ff':
        train_transform = base + [ssk, ff, ff_augs]
    elif transform == 'rsk':
        train_transform = base + [rsk, augs]
    elif transform == 'rsk_ff':
        train_transform = base + [rsk, ff, ff_augs]
    elif transform == 'ff':
        train_transform = base + [ff, ff_augs]
    else:
        train_transform = base  # clean

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose([transforms.ToTensor()])

    if normalize:
        train_transform.transforms.append(transforms.Normalize(mean, std))
        test_transform.transforms.append(transforms.Normalize(mean, std))

    return train_transform, test_transform
