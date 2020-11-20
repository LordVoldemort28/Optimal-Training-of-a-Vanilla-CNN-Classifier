import torch
import torchvision
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from scripts.transforms import resnet_augmentation, vgg_augmentation


def load_cifar10_dataset(configs):

    data_dir = configs.data_dir
    batch_size = configs.batch_size
    num_workers = configs.num_workers

    train_transform = vgg_augmentation()
    test_transform = vgg_augmentation(test=True)

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                     transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True,
                                    transform=test_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True
    )

    loader = {
        'train': train_loader,
        'test': test_loader
    }

    if configs.validation:
        valid_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                         transform=test_transform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(configs.valid_size * num_train))

        train_idx, valid_idx = indices[split:], indices[:split]

        if configs.shuffle:
            np.random.seed(configs.random_seed)
            np.random.shuffle(train_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=True
        )

        loader['train'], loader['valid'] = train_loader, valid_loader

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return loader, classes
