from torchvision import transforms


def resnet_augmentation(test=False):
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
