import torch
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import AverageMeter, get_filter_unlearnable
from resnet import resnet18 as net
from augments import get_transform

parser = argparse.ArgumentParser()

# training args
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--dataset',
                    type=str,
                    default='cifar10',
                    choices=[
                        'cifar10',
                        'cifar100',
                    ])
parser.add_argument('--normalize',
                    type=str,
                    default='yes',
                    choices=['yes', 'no'])
# unlearnable dataset args
parser.add_argument('--ud',
                    type=str,
                    default='cuda',
                    choices=['cuda', 'clean'])
# cuda args
parser.add_argument('--blur_parameter', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=0)
# percent of poisoned data, default 100% data is poisoned.
parser.add_argument('--mix', type=float, default=1.0)

# rsk + dct args
parser.add_argument('--transform',
                    type=str,
                    default='rsk_ff',
                    choices=['ssk', 'ssk_ff', 'rsk', 'rsk_ff', 'ff', 'clean'])
parser.add_argument('--sharp_center', type=float, default=2.5)
parser.add_argument('--ff_percentage', type=int, default=None)

args = parser.parse_args()

batch_size = 512
# transform args
normalize = True if args.normalize == 'yes' else False

# sets the number of classes, batch size, and mean/std depending on which dataset.
# mean and std are taken from mean and std of each dataset
if args.dataset == 'cifar10':
    num_cls = 10
else:
    num_cls = 100

train_transform, test_transform = get_transform(
    args.transform,
    args.dataset,
    normalize,
    args.sharp_center,
    args.ff_percentage,
)
if args.dataset == 'cifar10':
    train_dataset = datasets.CIFAR10(root='.',
                                     train=True,
                                     download=True,
                                     transform=train_transform)
    test_dataset = datasets.CIFAR10(root='.',
                                    train=False,
                                    download=True,
                                    transform=test_transform)
else:
    train_dataset = datasets.CIFAR100(root='.',
                                      train=True,
                                      download=True,
                                      transform=train_transform)
    test_dataset = datasets.CIFAR100(root='.',
                                     train=False,
                                     download=True,
                                     transform=test_transform)

if args.ud == 'cuda':
    print("Training on CUDA!")
    # To train on CUDA!
    cuda_ud, cnns = get_filter_unlearnable(args.blur_parameter, args.seed,
                                           num_cls, args.mix, args.dataset)

    cuda_dataset = cuda_ud
    cuda_dataset.transforms = train_dataset.transforms
    cuda_dataset.transform = train_dataset.transform

    train_loader = DataLoader(dataset=cuda_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=False,
                              num_workers=4)
else:
    print("Training Clean Dataset!")
    # to train on clean!
    train_loader = DataLoader(dataset=train_dataset,
                              shuffle=True,
                              num_workers=4,
                              batch_size=batch_size)

test_loader = DataLoader(dataset=test_dataset,
                         shuffle=True,
                         num_workers=4,
                         batch_size=batch_size)

# Preparing to Train

torch.manual_seed(args.seed)
model = net(3, num_cls).cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1,
                            weight_decay=0.0005,
                            momentum=0.9)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=args.epochs)

train_acc = []
test_acc = []

for epoch in range(args.epochs):
    # Train
    model.train()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    pbar = tqdm(train_loader, total=len(train_loader))

    for images, labels in pbar:

        images, labels = images.cuda(), labels.cuda()
        model.zero_grad()
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        _, predicted = torch.max(logits.data, 1)
        acc = (predicted == labels).sum().item() / labels.size(0)
        acc_meter.update(acc)
        loss_meter.update(loss.item())
        train_acc.append(acc)
        pbar.set_description("Train Acc %.2f Loss: %.2f" %
                             (acc_meter.avg * 100, loss_meter.avg))

    scheduler.step()

    # Eval
    model.eval()
    correct, total = 0, 0

    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    test_acc.append(acc)
    tqdm.write('Test Accuracy %.2f\n' % (acc * 100))
    tqdm.write('Epoch %.2f\n' % (epoch + 1))

print("Final Test Accuracy:", test_acc[-1])
