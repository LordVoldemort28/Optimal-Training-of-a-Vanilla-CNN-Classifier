from torchvision import transforms


def vgg_augmentation(test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if test == True:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])


def resnet_augmentation(test=False):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    if test == True:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
