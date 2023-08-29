"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import argparse
import time
import datetime
import json

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, accuracy_score

from models_interface.GPNN_Event import GPNN_Event
from datasets_interface.utils import load_best_checkpoint, save_checkpoint

from logutil import AverageMeter
from utils import get_event_data


def evaluation(pred_link_labels, link_labels, y_true, y_pred):
    # (batch_size, 5, link_num, 2), (batch_size, 5, link_num)
    pred_link_labels = pred_link_labels.permute(1, 0, 2, 3)
    link_labels = link_labels.permute(1, 0, 2)
    for class_idx in range(len(y_true)):
        y_pred[class_idx].extend(torch.argmax(pred_link_labels[class_idx].view(-1, 2), dim=1).tolist())
        y_true[class_idx].extend(link_labels[class_idx].view(-1).tolist())


def loss_fn(pred_link_labels, link_labels, cross_entropy_loss):
    # (batch_size, 5, link_num, 2), (batch_size, 5, link_num)
    pred_link_labels = pred_link_labels.permute(0, 3, 1, 2)
    loss = cross_entropy_loss(pred_link_labels, link_labels)
    return loss


def compute_metrics(y_true, y_pred):
    eval_rec = []
    eval_pre = []
    eval_f1 = []
    eval_acc = []
    for class_idx in range(5):
        eval_acc.append(accuracy_score(y_true[class_idx], y_pred[class_idx]))
        tn, fp, fn, tp = confusion_matrix(y_true[class_idx], y_pred[class_idx], labels=[0, 1]).ravel()
        tmp_rec = tp / (tp + fn) if tp != 0 else 0.
        eval_rec.append(tmp_rec)
        tmp_pre = tp / (tp + fp) if tp != 0 else 0.
        eval_pre.append(tmp_pre)
        tmp_f1 = 2 * tmp_rec * tmp_pre / (tmp_pre + tmp_rec) if tp != 0 else 0.
        eval_f1.append(tmp_f1)
    return eval_acc, eval_rec, eval_pre, eval_f1


def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    # logger = SummaryWriter(log_dir=os.path.join(args.output_root, "log", timestamp))

    # Load data
    training_set, valid_set, testing_set, train_loader, valid_loader, test_loader = get_event_data(args)

    # Get data size and define model
    edge_ids, node_features, link_labels, event_num = training_set[0]
    # link_labels: (batch_size, link_class_num, node_num * (node_num - 1) / 2, 2)

    node_feature_size = node_features.shape[1]
    edge_feature_size = 512
    link_class_num = link_labels.shape[1]
    model_args = {'model_path': args.resume, 'edge_feature_size': edge_feature_size,
                  'node_feature_size': node_feature_size,
                  'link_hidden_size': 512, 'link_hidden_layers': 2, 'link_relu': False,
                  'update_hidden_layers': 1, 'update_dropout': 0, 'update_bias': True,
                  'readout_hidden_size': 512,
                  'propagate_layers': 3, 'link_classes': link_class_num}
    model = GPNN_Event(model_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    model.to(args.device)
    cross_entropy_loss.to(args.device)

    best_epoch_f1 = -1
    loaded_checkpoint = load_best_checkpoint(args, model, optimizer)
    if loaded_checkpoint:
        args, best_epoch_f1, model, optimizer = loaded_checkpoint

    print("----- Start training -----")
    for epoch in range(args.start_epoch, args.epochs):
        logger.add_scalar('learning_rate', args.lr)
        # train for one epoch
        train(args, train_loader, model, cross_entropy_loss, optimizer, epoch, logger=logger)
        # test on validation set
        epoch_f1 = validate(args, valid_loader, model, cross_entropy_loss, logger=logger)

        is_best = False
        if epoch_f1 > best_epoch_f1:
            is_best = True
            best_epoch_f1 = epoch_f1
            print(" * Save Best_epoch_f1: {}".format(best_epoch_f1))
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                         'best_epoch_f1': best_epoch_f1, 'optimizer': optimizer.state_dict()},
                        is_best=is_best, directory=args.output_root)

    print("----- Start testing -----")
    # For testing
    # args.resume = args.output_root
    loaded_checkpoint = load_best_checkpoint(args, model, optimizer)
    if loaded_checkpoint:
        args, best_epoch_f1, model, optimizer = loaded_checkpoint
    gen_test_result(args, test_loader, model)

    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def train(args, train_loader, model, cross_entropy_loss, optimizer, epoch, logger=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    y_true = [[] for _ in range(5)]
    y_pred = [[] for _ in range(5)]

    # switch to train mode
    model.train()

    end_time = time.time()

    for i, (edge_ids, node_features, link_labels, event_num) in enumerate(train_loader):
        data_time.update(time.time() - end_time)
        optimizer.zero_grad()

        edge_ids = edge_ids.to(args.device)
        node_features = node_features.to(args.device)
        link_labels = link_labels.to(args.device)

        pred_link_labels = model(edge_ids, node_features, link_labels, event_num, args)
        # (batch_size, 5, link_num, 2)
        loss = loss_fn(pred_link_labels, link_labels, cross_entropy_loss)

        # Log and back propagate
        evaluation(pred_link_labels, link_labels, y_true, y_pred)

        losses.update(loss.item(), edge_ids.size()[0])
        loss.backward()
        optimizer.step()
        model.zero_grad()

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % args.log_interval == 0 and i != 0:
            eval_acc, eval_rec, eval_pre, eval_f1 = compute_metrics(y_true, y_pred)
            print('\nEpoch: [{}][{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                  'Acc {eval_acc}\n'
                  'Rec {eval_rec}\n'
                  'Pre {eval_pre}\n'
                  'F1 {eval_f1}'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                          eval_acc=eval_acc, eval_rec=eval_rec, eval_pre=eval_pre, eval_f1=eval_f1))

    eval_acc, eval_rec, eval_pre, eval_f1 = compute_metrics(y_true, y_pred)
    avg_f1 = np.mean(eval_f1)

    if logger is not None:
        logger.add_scalar('train_epoch_loss', losses.avg)
        logger.add_scalar('train_epoch_f1', avg_f1)

    print('\nEpoch: [{0}]; Avg F1 {f1:.4f}; Avg Loss {loss.avg:.4f}; Avg Time x Batch {b_time.avg:.4f}'
          .format(epoch, f1=avg_f1, loss=losses, b_time=batch_time))


def validate(args, val_loader, model, cross_entropy_loss, logger=None):
    losses = AverageMeter()

    y_true = [[] for _ in range(5)]
    y_pred = [[] for _ in range(5)]

    # switch to evaluate mode
    model.eval()

    for i, (edge_ids, node_features, link_labels, event_num) in enumerate(val_loader):
        edge_ids = edge_ids.to(args.device)
        node_features = node_features.to(args.device)
        link_labels = link_labels.to(args.device)

        pred_link_labels = model(edge_ids, node_features, link_labels, event_num, args)
        loss = loss_fn(pred_link_labels, link_labels, cross_entropy_loss)

        # Log
        losses.update(loss.item(), edge_ids.size()[0])
        evaluation(pred_link_labels, link_labels, y_true, y_pred)

    eval_acc, eval_rec, eval_pre, eval_f1 = compute_metrics(y_true, y_pred)
    avg_f1 = np.mean(eval_f1)

    print('\n* Eval: Avg F1 {f1:.4f}\t'
          'Avg Loss {loss.avg:.4f}\n'
          'Eval_Acc {eval_acc}\n'
          'Eval_Rec {eval_rec}\n'
          'Eval_Pre {eval_pre}\n'
          'Eval_F1 {eval_f1}'
          .format(f1=avg_f1, loss=losses,
                  eval_acc=eval_acc, eval_rec=eval_rec, eval_pre=eval_pre, eval_f1=eval_f1))

    if logger is not None:
        logger.add_scalar('val_epoch_loss', losses.avg)
        logger.add_scalar('val_epoch_f1', avg_f1)

    return avg_f1


def gen_test_result(args, test_loader, model):
    # switch to evaluate mode
    model.eval()

    y_true = [[] for _ in range(5)]
    y_pred = [[] for _ in range(5)]
    all_results = []
    for i, (edge_ids, node_features, link_labels, event_num) in enumerate(test_loader):
        edge_ids = edge_ids.to(args.device)
        node_features = node_features.to(args.device)
        link_labels = link_labels.to(args.device)

        pred_link_labels = model(edge_ids, node_features, link_labels, event_num, args)
        # (batch_size, 5, link_num, 2)
        pred_result = torch.argmax(pred_link_labels, dim=3).tolist()
        all_results.extend(pred_result)

        evaluation(pred_link_labels, link_labels, y_true, y_pred)

    eval_acc, eval_rec, eval_pre, eval_f1 = compute_metrics(y_true, y_pred)
    avg_f1 = np.mean(eval_f1)
    print('* Test: Avg F1 {f1:.4f}\n'
          'Eval_Acc {eval_acc}\n'
          'Eval_Rec {eval_rec}\n'
          'Eval_Pre {eval_pre}\n'
          'Eval_F1 {eval_f1}'
          .format(f1=avg_f1,
                  eval_acc=eval_acc, eval_rec=eval_rec, eval_pre=eval_pre, eval_f1=eval_f1))

    test_out_path = os.path.join(args.output_root, "test_result.json")
    with open(test_out_path, "w") as f_out:
        json.dump(all_results, f_out)


def parse_arguments():
    # Parser check
    def restricted_float(x, inter):
        x = float(x)
        if x < inter[0] or x > inter[1]:
            raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]" % (x,))
        return x

    # Path settings
    parser = argparse.ArgumentParser(description='EventGraph Prediction')
    parser.add_argument('--data-root', required=True, help='data path')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--resume', default=None)

    # Optimization Options
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='Input batch size for training (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='Index of epoch to start (default: 0)')
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-5, metavar='LR',
                        help='Initial learning rate [1e-5, 1e-2] (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    # i/o
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='How many batches to wait before logging training status')
    # Accelerating
    parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    print("Training args...")
    print(args)
    main(args)
