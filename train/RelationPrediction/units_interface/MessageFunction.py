"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch
import torch.nn
import torch.autograd

from argparse import Namespace


class MessageFunctionForEvent(torch.nn.Module):
    def __init__(self, args: Namespace):
        super(MessageFunctionForEvent, self).__init__()
        self.args = args

        self.learn_modules = torch.nn.ModuleList([])
        edge_feature_size = self.args['edge_feature_size']
        node_feature_size = self.args['node_feature_size']
        message_size = self.args['edge_feature_size']

        self.e2m = torch.nn.Linear(edge_feature_size, message_size, bias=True)
        self.n2m = torch.nn.Linear(node_feature_size, message_size, bias=True)
        self.resize_m = torch.nn.Linear(message_size * 3, message_size, bias=True)

    # Message from h_w to h_v through e_vw
    def forward(self, h_w, h_v, e_wv, args=None):
        # (1, node_size, node_num), (1, node_size), (1, edge_size, node_num)
        message = torch.zeros(e_wv.size()[0], self.args['edge_feature_size'], e_wv.size()[2]).to(args.device)

        for i_node in range(e_wv.size()[2]):
            tmp_msg = torch.cat([self.e2m(e_wv[:, :, i_node]), self.n2m(h_w[:, :, i_node]), self.n2m(h_v)], 1)
            # (1, 3 * message_size)
            message[:, :, i_node] = self.resize_m(tmp_msg)

        return message
        # (1, message_size, valid_node_num)

    # Get the message function arguments
    def get_args(self):
        return self.args


def main():
    pass


if __name__ == '__main__':
    main()
