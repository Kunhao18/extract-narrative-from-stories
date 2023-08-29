"""
Created on Oct 03, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch
import torch.nn


class LinkFunctionForEvent(torch.nn.Module):
    def __init__(self, args: dict):
        super(LinkFunctionForEvent, self).__init__()
        self.args = args
        input_size = self.args['edge_feature_size'] + 2 * self.args['node_feature_size']
        hidden_size = self.args['link_hidden_size']

        self.learn_modules = torch.nn.ModuleList([])
        for _ in range(self.args['link_hidden_layers'] - 1):
            self.learn_modules.append(torch.nn.Conv2d(input_size, hidden_size, 1))
            self.learn_modules.append(torch.nn.ReLU())
            input_size = hidden_size
        self.learn_modules.append(torch.nn.Conv2d(input_size, 1, 1))

    def forward(self, edge_features, node_features):
        # (1, edge_size, node_num, node_num), (1, node_size, node_num)
        node_num = node_features.size()[2]
        node_addon_row = node_features.unsqueeze(3).repeat(1, 1, 1, node_num)
        node_addon_col = node_features.unsqueeze(2).repeat(1, 1, node_num, 1)
        last_layer_output = torch.cat([edge_features, node_addon_row, node_addon_col], 1)
        # (1, edge_size + 2 * node_size, node_num, node_num)
        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        return last_layer_output[:, 0, :, :]

    def get_args(self):
        return self.args


def main():
    pass


if __name__ == '__main__':
    main()
