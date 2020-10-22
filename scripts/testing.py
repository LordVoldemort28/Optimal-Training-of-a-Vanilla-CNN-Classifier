import torch
import random
import numpy as np
from sklearn.metrics import confusion_matrix


def test(configs):

    model = configs.model
    loader = configs.loader
    num_classes = configs.num_classes
    criterion = configs.criterion.cuda()

    correct = 0.
    test_loss = 0.
    total = 0.

    confusion_matrix = torch.zeros(num_classes, num_classes)

    model.eval()

    with configs.experiment.test():
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader['test']):
                print("Current batch: {}".format(batch_idx), end="\r")

                outputs = model(data)

                loss = criterion(outputs, target)

                test_loss += ((1 / (batch_idx + 1)) *
                              (loss.data - test_loss)).cpu().numpy()

                # get prediction
                pred = outputs.data.max(1, keepdim=True)[1]

                # populate the confusion matrix
                for t, p in zip(target.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                    # accumulate correct predictions.
                correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

                # accumulate total number of examples.
                total += data.size(0)

                configs.experiment.log_metric(
                    "accuracy", (correct/total), step=batch_idx+1)
    # Print Total Testing Loss and Accuracy
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    # log confusion matrix in comet.ml
    configs.experiment.log_confusion_matrix(
        labels=configs.labels, max_categories=100, matrix=confusion_matrix)
