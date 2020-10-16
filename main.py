from comet_ml import Experiment
import os
import torch
import torch.nn as nn

from models.vanillaCNN import Net
from utils.helpers import config_dict, Config, get_optimizer
from scripts.training import train
from scripts.loaders import load_cifar10_dataset
from scripts.criterions import cross_entropy_loss


def initialization(configs):
    configs.loader, configs.labels = load_cifar10_dataset(configs)
    
    if configs.activation_function != "adam":
        model = Net(activation_function=configs.activation_function)
    else:
        model = Net()
        
    if torch.cuda.is_available() == True:
        model = nn.DataParallel(model)
        configs.model = model
    else:
        print("Please run the experiment in gpu")
        exit(1)

    configs.criterion = cross_entropy_loss()
    configs.optimizer = get_optimizer(configs)

    train(configs)


if __name__ == "__main__":
    params_dict = config_dict(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'configs.txt'))
    configs = Config(params_dict)

    # Start comet ML Experiment
    experiment = Experiment(api_key="W3E2kwOfZBiw21K5tJdkmxomM",
                            project_name="learning-deep", workspace="lordvoldemort28")
    experiment.set_name(configs.experiment_name)
    experiment.add_tag("initialization")

    # Log hyperparameters in comet ML
    experiment.log_parameters(configs)

    configs.experiment = experiment

    initialization(configs)

    experiment.end()
