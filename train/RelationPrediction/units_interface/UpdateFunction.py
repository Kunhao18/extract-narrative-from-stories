"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch


class UpdateFunctionForEvent(torch.nn.Module):
    def __init__(self, args: dict):
        super(UpdateFunctionForEvent, self).__init__()
        self.args = args
        self.learn_modules = torch.nn.ModuleList([])

        input_feature_size = self.args['node_feature_size']
        message_size = self.args['edge_feature_size']
        num_layers = self.args.get('update_hidden_layers', 1)
        bias = self.args.get('update_bias', False)
        dropout = self.args.get('update_dropout', 0)

        self.update_gru = torch.nn.GRU(message_size, input_feature_size,
                                       num_layers=num_layers, bias=bias, dropout=dropout)

    def forward(self, h_v, m_v, args=None):
        # (1, 1, input_size), (1, 1, edge_size)
        output, h = self.update_gru(m_v, h_v)
        # (1, 1, input_size), (1, 1, input_size)
        return output

    def get_args(self):
        return self.args


class UpdateFunctionForEventLink(torch.nn.Module):
    def __init__(self, args: dict):
        super(UpdateFunctionForEventLink, self).__init__()
        self.args = args
        self.learn_modules = torch.nn.ModuleList([])

        input_feature_size = self.args['edge_feature_size']
        message_size = self.args['edge_feature_size']
        num_layers = self.args.get('update_hidden_layers', 1)
        bias = self.args.get('update_bias', False)
        dropout = self.args.get('update_dropout', 0)

        self.update_gru = torch.nn.GRU(message_size, input_feature_size,
                                       num_layers=num_layers, bias=bias, dropout=dropout)

    def forward(self, e_wv, m_wv, args=None):
        # (1, input_size, valid_node_num), (1, edge_size, valid_node_num)
        output = torch.zeros_like(e_wv).to(args.device)
        for i_node in range(e_wv.size()[2]):
            tmp_output, h = self.update_gru(m_wv[:, :, i_node][None], e_wv[:, :, i_node][None])
            # (1, 1, input_size), (1, 1, input_size)
            output[:, :, i_node] = tmp_output.squeeze(0)
        return output

    def get_args(self):
        return self.args


def main():
    pass


if __name__ == '__main__':
    main()
