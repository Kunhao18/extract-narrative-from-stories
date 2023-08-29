"""
Created on Oct 04, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import shutil

import numpy as np
import torch


def collate_fn_event(batch):
    edge_ids, node_features, link_labels, event_num = batch[0]
    max_event_num = np.max(np.array([event_num for (edge_ids, node_features, link_labels, event_num) in batch],
                                    dtype=int))
    max_link_num = int(max_event_num * (max_event_num - 1) / 2)

    node_feature_len = node_features.shape[1]
    edge_ids_batch = np.zeros((len(batch), max_event_num, max_event_num), dtype=int)
    node_features_batch = np.zeros((len(batch), max_event_num, node_feature_len))
    link_labels_batch = np.full((len(batch), 5, max_link_num), fill_value=-1, dtype=int)
    event_nums_batch = list()

    for i, (edge_ids, node_features, link_labels, event_num) in enumerate(batch):
        link_num = int(event_num * (event_num - 1) / 2)
        edge_ids_batch[i, :event_num, :event_num] = edge_ids
        node_features_batch[i, :event_num, :] = node_features
        link_labels_batch[i, :, :link_num] = link_labels
        event_nums_batch.append(event_num)

    edge_ids_batch = torch.LongTensor(edge_ids_batch)
    node_features_batch = torch.FloatTensor(node_features_batch)
    link_labels_batch = torch.LongTensor(link_labels_batch)

    return edge_ids_batch, node_features_batch, link_labels_batch, event_nums_batch


def save_checkpoint(state, is_best, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)


def load_best_checkpoint(args, model, optimizer):
    # get the best checkpoint if available without training
    if args.resume:
        print("loading from {}...".format(args.resume))
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_epoch_f1 = checkpoint['best_epoch_f1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.to(args.device)
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
            return args, best_epoch_f1, model, optimizer
        else:
            print("=> no best model found at '{}'".format(best_model_file))
    return None


def main():
    pass


if __name__ == '__main__':
    main()
