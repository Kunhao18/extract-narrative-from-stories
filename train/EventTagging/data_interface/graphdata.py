# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""AMR dataset."""


from inspect import EndOfBlock
import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """
PubMed articles.

There are three features:
  - src: source text.
  - tgt: AMR Graph.
"""


_SRC = "sent"
_TGT = "tree"


class EventGraphData(datasets.GeneratorBasedBuilder):

    # Version 1.2.0 expands coverage, includes ids, and removes web contents.
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _SRC: datasets.Value("string"),
                    _TGT: datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        test_path = self.config.data_files["test"]
        return [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        logger.info("generating examples from = %s", filepath[0])
        with open(filepath[0], "r", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                json_dict = json.loads(line)
                sent_list = []
                tree_list = []
                events = json_dict["events"]
                for event in events:
                    cur_text = event["sent"]
                    cur_tree = event["tree"]
                    sent_list.append(cur_text)
                    tree_list.append(cur_tree)
                src = "\1".join(sent_list)
                tgt = "\1".join(tree_list)

                yield idx, {_SRC: src, _TGT: tgt}
