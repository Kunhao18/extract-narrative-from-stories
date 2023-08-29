"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import itertools

import numpy as np
import torch
import torch.utils.data
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt

from datasets_interface.Event.event import EventData
from datasets_interface.utils import collate_fn_event


def plot_confusion_matrix(cm, classes, filename=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    ax = plt.gca()
    ax.tick_params(axis=u'both', which=u'both', length=0)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, '{0:.2f}'.format(cm[i, j]), verticalalignment='center', horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def get_event_data(args):
    np.random.seed(0)
    with open(os.path.join(args.data_root, 'train.txt')) as f:
        train_filenames = [line.strip() for line in f.readlines()]
    with open(os.path.join(args.data_root, 'val.txt')) as f:
        val_filenames = [line.strip() for line in f.readlines()]
    with open(os.path.join(args.data_root, 'test.txt')) as f:
        test_filenames = [line.strip() for line in f.readlines()]
    feature_root = os.path.join(args.data_root, "graph_features")

    training_set = EventData(feature_root, train_filenames[:])
    valid_set = EventData(feature_root, val_filenames[:])
    testing_set = EventData(feature_root, test_filenames[:])

    train_loader = torch.utils.data.DataLoader(training_set, collate_fn=collate_fn_event,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, collate_fn=collate_fn_event,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testing_set, collate_fn=collate_fn_event,
                                              batch_size=args.batch_size,
                                              num_workers=args.prefetch, pin_memory=True)
    print('Dataset sizes: {} training, {} validation, {} testing.'.format(len(train_loader), len(valid_loader),
                                                                          len(test_loader)))
    return training_set, valid_set, testing_set, train_loader, valid_loader, test_loader


def main():
    pass


if __name__ == '__main__':
    main()
