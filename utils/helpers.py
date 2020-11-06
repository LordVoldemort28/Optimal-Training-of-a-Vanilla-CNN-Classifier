import os
import time
import torch.nn as nn
import torch
from datetime import datetime, timedelta
from scripts.optimizers import adam_optimizer, sgd_optimizer
from models.vanillaCNN import Net
from models import vgg
from models import resnet


class Config(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def get_time_format(d):
    if d.day - 1 == 1:
        return "%.2d:%.2d:%.2d:%.2d" % (d.day - 1, d.hour, d.minute, d.second)
    return "%.2d:%.2d:%.2d" % (d.hour, d.minute, d.second)


def calculate_time(start_time, epoch_duration=False):
    cal_time = time.time() - start_time
    sec = timedelta(seconds=int(cal_time))
    d = datetime(1, 1, 1) + sec
    if epoch_duration:
        return sec.seconds
    return get_time_format(d)


def castType(s):
    """ Function casts the string value to the right type
    for the hyperparameters"""

    s = str(s.replace(" ", ""))

    try:
        s = int(s)
    except ValueError:
        try:
            s = float(s)
        except ValueError:
            s = str(s)

    return s


def get_optimizer(configs):
    if configs.optimizer == 'adam':
        return adam_optimizer(configs)
    elif configs.optimizer == 'sgd':
        return sgd_optimizer(configs)


def load_model(configs):
    if configs.model == 'vanilla':
        if configs.activation_function != "relu":
            return Net(activation_function=configs.activation_function)
        else:
            return Net()
    elif configs.model == 'vgg':
        model = vgg.__dict__[configs.vgg_model]()
        model.features = torch.nn.DataParallel(model.features)
        return model
    elif configs.model == 'resnet':
        model = resnet.__dict__[configs.resnet_model]()
        model = torch.nn.DataParallel(model)
        return model
    else:
        print("Please config model")
        exit(1)


def config_dict(location):
    """
    Function reads the configs.txt file and returns
    a dictionary with all the hyperparameters key, value
    pairs. 
    """

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    file = open(os.path.join(__location__, location))
    parameters = file.readlines()

    params = {}
    for parameter in parameters:
        # remove whitespace
        parameter = parameter.replace(" ", "")
        parameter = parameter.replace("\n", "")
        # get the key and value
        key, value = parameter.split("=")

        # convert value to the right type
        value = castType(value)

        params[key] = value

    return params


def get_activation_function(configs):
    if configs.activation == 'tanh':
        return nn.Tanh
    elif configs.activation == 'sigmod':
        return nn.Sigmoid
    else:
        return nn.ReLU
