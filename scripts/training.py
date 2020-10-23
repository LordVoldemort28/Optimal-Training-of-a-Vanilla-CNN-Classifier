import time
import torch
from utils.helpers import calculate_time
import numpy as np


def train(configs):

    epochs = configs.epochs
    model = configs.model
    loader = configs.loader
    optimizer = configs.optimizer
    criterion = configs.criterion.cuda()

    running_loss = 0
    steps = 0

    for epoch in range(1, epochs+1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        correct = 0.
        total = 0.
        epoch_time = time.time()

        adjust_learning_rate(configs, optimizer, epoch)

        print("============Epoch: {}=============".format(epoch))
        with configs.experiment.train():

            for batch_idx, (data, target) in enumerate(loader['train']):
                steps += 1

                if torch.cuda.is_available() == True:
                    data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()

                # get output
                output = model(data)

                # calculate loss and backprop
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # record the average training loss
                train_loss += ((1 / (batch_idx + 1)) *
                               (loss.data - train_loss)).cpu().numpy()

                # convert output prop to predicted class
                pred = output.data.max(1, keepdim=True)[1]

                # accumulate correct predictions.
                correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

                # accumulate total number of examples.
                total += data.size(0)
            # compute training accuracy.
            train_acc = correct / total
            print("train_acc = {}".format(train_acc))

            configs.experiment.log_metric(
                'accuracy', train_acc, epoch=epoch)
            configs.experiment.log_metric(
                'loss', float(train_loss), epoch=epoch)

        # validate model
        model.eval()
        correct = 0.
        total = 0.

        step_val = 0
        with configs.experiment.validate():
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(loader['valid']):

                    if torch.cuda.is_available() == True:
                        data, target = data.cuda(), target.cuda()

                    # Get output
                    output = model(data)
                    # Calculate loss
                    loss = criterion(output, target)

                    pred = output.data.max(1, keepdim=True)[1]

                    # records the average validation loss
                    valid_loss += ((1/(batch_idx+1)) *
                                   (loss.data - valid_loss)).cpu().numpy()

                    # accumulate correct predictions
                    batch_correct_val = np.sum(np.squeeze(
                        pred.eq(target.data.view_as(pred))).cpu().numpy())

                    correct += batch_correct_val

                    # accumulate total number of examples
                    batch_total_val = data.size(0)
                    total += batch_total_val

            valid_acc = correct / total
            print("valid_acc = {}".format(valid_acc))
            print("Epoch duration: {}\n".format(calculate_time(epoch_time)))

            configs.experiment.log_metric(
                'accuracy', valid_acc, epoch=epoch
            )
            configs.experiment.log_metric(
                'loss', float(valid_loss), epoch=epoch
            )
            configs.experiment.log_metric(
                'epoch duration', epoch_time, epoch=epoch
            )


def adjust_learning_rate(configs, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = configs.learning_rate * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
