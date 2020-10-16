import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, activation_function=nn.ReLU):
        super(Net, self).__init__()
        self.conv1 = self.conv_layer(in_channels=3, out_channels=32)
        self.conv2 = self.conv_layer(
            in_channels=32, out_channels=32, pool_size=2, dropout_rate=0.2)
        self.conv3 = self.conv_layer(in_channels=32, out_channels=64)
        self.conv4 = self.conv_layer(
            in_channels=64, out_channels=64, pool_size=2, dropout_rate=0.3)
        self.conv5 = self.conv_layer(in_channels=64, out_channels=128)
        self.conv6 = self.conv_layer(
            in_channels=128, out_channels=128, pool_size=2, dropout_rate=0.4)
        self.flatten = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=128*4*4, out_features=128),
            nn.BatchNorm1d(128),
            activation_function(True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=10)
        )

    def conv_layer(self, in_channels, out_channels, kernal_size=3, stride=1, pool_size=None, dropout_rate=None, padding=(1, 1), activation_function=nn.ReLU):
        layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernal_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            activation_function(True))

        if dropout_rate or pool_size:
            layers.add_module("3", nn.MaxPool2d(pool_size))
            layers.add_module("4", nn.Dropout2d(dropout_rate))

        return layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        return nn.functional.log_softmax(x, dim=1)
