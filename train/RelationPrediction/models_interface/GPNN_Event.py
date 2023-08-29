"""
Created on Oct 07, 2017

@author: Siyuan Qi

Description of the file.

"""

import os

import torch
import torch.nn
import torch.autograd

from units_interface.LinkFunction import LinkFunctionForEvent
from units_interface.MessageFunction import MessageFunctionForEvent
from units_interface.UpdateFunction import UpdateFunctionForEvent, UpdateFunctionForEventLink
from units_interface.ReadoutFunction import ReadoutFunctionForEvent


class GPNN_Event(torch.nn.Module):
    def __init__(self, model_args):
        super(GPNN_Event, self).__init__()

        self.model_args = model_args.copy()

        self.edge_embedding = torch.nn.Embedding(4, model_args['edge_feature_size'], padding_idx=0)
        self.link_fun = LinkFunctionForEvent(model_args)
        self.sigmoid = torch.nn.Sigmoid()
        self.message_fun = MessageFunctionForEvent(model_args)
        self.update_fun = UpdateFunctionForEvent(model_args)
        self.update_link_fun = UpdateFunctionForEventLink(model_args)
        self.readout_fun = ReadoutFunctionForEvent({'readout_input_size': model_args['edge_feature_size'] * 2,
                                                    'hidden_size': model_args['readout_hidden_size']})

        self.propagate_layers = model_args['propagate_layers']

        self._load_link_fun(model_args)

    def forward(self, edge_ids, node_features, link_labels, event_nums, args):
        # TODO: correct the edge_ids size
        # edge_ids: (batch_size, node_num, node_num, 1)
        edge_ids = edge_ids.squeeze(-1)
        # edge_ids: (batch_size, node_num, node_num)
        edge_features = self.edge_embedding(edge_ids)
        # edge_features: (batch_size, node_num, node_num, edge_size)

        node_num = edge_features.size()[1]
        batch_size = edge_features.size()[0]
        edge_features = edge_features.permute(0, 3, 1, 2)
        # edge_features: (batch_size, edge_size, node_num, node_num)
        node_features = node_features.permute(0, 2, 1)
        # node_features: (batch_size, node_size, node_num)
        hidden_node_states = [[node_features[batch_i, ...].unsqueeze(0).clone()
                               for _ in range(self.propagate_layers + 1)]
                              for batch_i in range(node_features.size()[0])]
        # (batch_size, propagate_steps + 1, 1, feature_size, node_num)
        hidden_edge_states = [[edge_features[batch_i, ...].unsqueeze(0).clone()
                               for _ in range(self.propagate_layers + 1)]
                              for batch_i in range(node_features.size()[0])]
        # (batch_size, propagate_steps + 1, 1, feature_size, node_num, node_num)

        pred_adj_mat = torch.zeros([batch_size, node_num, node_num]).to(args.device)
        pred_link_labels = torch.zeros([batch_size, 5, link_labels.size()[2], 2]).to(args.device)

        # Belief propagation
        for batch_idx in range(node_features.size()[0]):
            valid_node_num = event_nums[batch_idx]

            for passing_round in range(self.propagate_layers):
                # print hidden_edge_states[batch_idx][passing_round].size(), valid_node_num
                pred_adj_mat[batch_idx, :valid_node_num, :valid_node_num] = \
                    self.link_fun(hidden_edge_states[batch_idx][passing_round][:, :, :valid_node_num, :valid_node_num],
                                  hidden_node_states[batch_idx][passing_round][:, :, :valid_node_num])
                sigmoid_pred_adj_mat = self.sigmoid(pred_adj_mat[batch_idx, :, :]).unsqueeze(0)
                # (1, valid_node_num, valid_node_num)

                # Loop through nodes
                for i_node in range(valid_node_num):
                    h_v = hidden_node_states[batch_idx][passing_round][:, :, i_node]
                    # (1, node_size)
                    h_w = hidden_node_states[batch_idx][passing_round][:, :, :valid_node_num]
                    # (1, node_size, node_num)
                    e_wv = hidden_edge_states[batch_idx][passing_round][:, :, :valid_node_num, i_node]
                    # e_wv = edge_features[batch_idx, :, :valid_node_num, i_node].unsqueeze(0)
                    # (1, edge_size, node_num)
                    m_v = self.message_fun(h_w, h_v, e_wv, args)  # h_w to h_v
                    # (1, edge_size, valid_node_num)

                    # Sum up messages from different nodes according to weights
                    m_v = sigmoid_pred_adj_mat[:, :valid_node_num, i_node].unsqueeze(1).expand_as(m_v) * m_v
                    # (1, 1, valid_node_num) * (1, edge_size, valid_node_num)
                    e_wv = self.update_link_fun(e_wv, m_v, args)
                    # (1, edge_size, valid_node_num)
                    hidden_edge_states[batch_idx][passing_round + 1][:, :, :valid_node_num, i_node] = e_wv
                    # (1, edge_size, valid_node_num)
                    m_v = torch.sum(m_v, 2)
                    # (1, edge_size)
                    h_v = self.update_fun(h_v[None].contiguous(), m_v[None], args)
                    # (1, 1, node_size)
                    hidden_node_states[batch_idx][passing_round + 1][:, :, i_node] = h_v.squeeze(0)

                if passing_round == self.propagate_layers - 1:
                    pred_link_idx = 0
                    for i_node in range(valid_node_num):
                        for j_node in range(i_node + 1, valid_node_num):
                            pred_link_labels[batch_idx, :, pred_link_idx, :] = self.readout_fun(
                                hidden_edge_states[batch_idx][passing_round + 1][:, :, i_node, j_node],
                                # (1, edge_size)
                                hidden_edge_states[batch_idx][passing_round + 1][:, :, j_node, i_node]
                                # (1, edge_size)
                            )
                            pred_link_idx += 1
                # link_labels: (batch_size, link_class_num, node_num * (node_num - 1) / 2, 2)

        return pred_link_labels

    def _load_link_fun(self, model_args):
        if model_args['model_path'] is None:
            return
        if not os.path.exists(model_args['model_path']):
            return
        best_model_file = os.path.join(model_args['model_path'], os.pardir, 'graph', 'model_best.pth')
        if os.path.isfile(best_model_file):
            checkpoint = torch.load(best_model_file)
            self.link_fun.load_state_dict(checkpoint['state_dict'])


def main():
    pass


if __name__ == '__main__':
    main()
