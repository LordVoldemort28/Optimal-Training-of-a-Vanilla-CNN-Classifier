import torch


def adam_optimizer(configs):
    return torch.optim.Adam(
        configs.model.parameters(),
        lr=configs.learning_rate,
    )


def sgd_optimizer(configs):
    return torch.optim.SGD(
        configs.model.parameters(),
        lr=configs.learning_rate,
        momentum=configs.momentum,
        weight_decay=configs.weight_decay
    )
