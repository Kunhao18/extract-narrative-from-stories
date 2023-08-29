"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch


class ReadoutFunctionForEvent(torch.nn.Module):
    def __init__(self, args: dict):
        super(ReadoutFunctionForEvent, self).__init__()
        self.args = args
        input_size = self.args['readout_input_size']
        hidden_size = self.args['hidden_size']

        self.learn_modules_fc1 = torch.nn.ModuleList([])
        self.learn_modules_fc2 = torch.nn.ModuleList([])
        for _ in range(5):
            self.learn_modules_fc1.append(torch.nn.Linear(input_size, hidden_size))
        for _ in range(5):
            self.learn_modules_fc2.append(torch.nn.Linear(hidden_size, 2))

    def forward(self, edge_features_i, edge_features_j):
        edge_features = torch.cat([edge_features_i, edge_features_j], 1)
        logits = torch.zeros([5, 2])
        for class_idx in range(5):
            tmp_hidden = self.learn_modules_fc1[class_idx](edge_features)
            # (1, hidden_size)
            logits[class_idx, :] = self.learn_modules_fc2[class_idx](tmp_hidden).squeeze(0)
            # (2)
        return logits

    def get_args(self):
        return self.args


def main():
    pass


if __name__ == '__main__':
    main()
