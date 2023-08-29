import os
import time
import json
import argparse

import numpy as np
import torch.utils.data


class EventData(torch.utils.data.Dataset):
    def __init__(self, root, sequence_ids):
        self.root = root
        self.sequence_ids = sequence_ids

    def __getitem__(self, index):
        sequence_id = self.sequence_ids[index]
        data = json.load(open(os.path.join(self.root, 'info_{}.json'.format(sequence_id)), 'r'))

        event_num = data['event_num']
        link_labels = data['link_labels']
        link_labels = np.array(link_labels, dtype=int)

        edge_ids = np.load(os.path.join(self.root, 'edge_id_{}.npy').format(sequence_id))
        node_features = np.load(os.path.join(self.root, 'node_feature_{}.npy').format(sequence_id))

        return edge_ids, node_features, link_labels, event_num

    def __len__(self):
        return len(self.sequence_ids)
