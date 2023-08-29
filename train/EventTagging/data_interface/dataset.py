# coding:utf-8
import os
import torch
from datasets import load_dataset
from dataclasses import dataclass
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


class EventTaggingDataSet(torch.nn.Module):
    def __init__(
            self,
            tokenizer,
            story_tokenizer,
            train_file,
            validation_file,
            test_file,
            prefix="",
            pad_to_max_length=True,
            max_src_length=512,
            max_tgt_length=512,
            ignore_pad_token_for_loss=True,
    ):
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.story_tokenizer = story_tokenizer
        self.pad_to_max_length = pad_to_max_length
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

    def setup(self):
        data_files = {"train": self.train_file, "validation": self.validation_file, "test": self.test_file}
        datasets = load_dataset(f"{os.path.dirname(__file__)}/taggingdata.py", data_files=data_files,
                                keep_in_memory=True)
        print("datasets:", datasets)
        column_names = datasets["train"].column_names
        print("colums:", column_names)
        padding = "max_length" if self.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            sources = examples["src"]  # AMR tokens
            labels = examples["label"]  # text tokens

            # print("!!!type ", type(sources))
            all_sents = [sent.split("\2")[0] for sent in sources.split("\1")]
            all_events = [sent.split("\2")[1].split("\3") for sent in sources.split("\1")]

            story_ids = []
            story_type_ids = []
            if self.story_tokenizer:
                story_text = " [SEP] ".join(all_sents)
                story_tokenized = self.story_tokenizer([story_text], max_length=512, truncation=True)
                story_ids = story_tokenized["input_ids"][0]
                cur_type = 0
                for cur_id in story_ids:
                    story_type_ids.append(cur_type)
                    if cur_id == self.story_tokenizer.sep_token_id:
                        cur_type = 1 - cur_type

            tmp_model_inputs = self.tokenizer(all_sents,
                                              max_length=self.max_src_length,
                                              padding=False,
                                              truncation=True)
            all_tokenized_sents = [
                srci
                + [self.tokenizer.amr_bos_token_id,
                   self.tokenizer.mask_token_id,
                   self.tokenizer.amr_eos_token_id]
                for srci in tmp_model_inputs["input_ids"]
            ]

            all_tokenized_amrs = []
            for sent_events in all_events:
                tmp_tokenized_amrs = [
                    [self.tokenizer.bos_token_id,
                     self.tokenizer.mask_token_id,
                     self.tokenizer.eos_token_id,
                     self.tokenizer.amr_bos_token_id]
                    + self.tokenizer.tokenize_amr(event.split())[:self.max_src_length - 1]
                    + [self.tokenizer.amr_eos_token_id]
                    for event in sent_events
                ]  # [<s> [mask] <\s> <AMR> y1,y2,...ym </AMR>]
                all_tokenized_amrs.append(tmp_tokenized_amrs)

            model_inputs = {"input_ids": story_ids,
                            "input_type_ids": story_type_ids,
                            "sent_ids": [],
                            "labels": [int(token_label) for token_label in labels.split()]}

            for sent_idx in range(len(all_tokenized_sents)):
                current_sent_input_dict = {"tokens": all_tokenized_sents[sent_idx],
                                           "events": all_tokenized_amrs[sent_idx]}
                model_inputs["sent_ids"].append(current_sent_input_dict)

            return model_inputs

        self.train_dataset = datasets["train"].map(tokenize_function,
                                                   batched=False, remove_columns=["src", "label"], num_proc=8)
        print(f"ALL {len(self.train_dataset)} training instances")
        self.valid_dataset = datasets["validation"].map(tokenize_function,
                                                        batched=False, remove_columns=["src", "label"], num_proc=8)
        print(f"ALL {len(self.valid_dataset)} validation instances")
        self.test_dataset = datasets["test"].map(tokenize_function,
                                                 batched=False, remove_columns=["src", "label"], num_proc=8)
        print(f"ALL {len(self.test_dataset)} test instances")

        # print("Dataset Instance Example:", self.train_dataset[0])


class EventGraphDataSet(torch.nn.Module):
    def __init__(self,
                 tokenizer,
                 test_file,
                 prefix="",
                 pad_to_max_length=True,
                 max_src_length=512,
                 max_tgt_length=512,
                 ignore_pad_token_for_loss=True):
        super().__init__()
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.pad_to_max_length = pad_to_max_length
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

    def setup(self):
        data_files = {"test": self.test_file}
        datasets = load_dataset(f"{os.path.dirname(__file__)}/graphdata.py", data_files=data_files,
                                keep_in_memory=True)
        print("datasets:", datasets)
        column_names = datasets["test"].column_names
        print("colums:", column_names)

        def tokenize_function(examples):
            # Remove empty lines
            sents = examples["sent"]  # AMR tokens
            trees = examples["tree"]  # text tokens

            # print("!!!type ", type(sources))
            all_sents = [sent for sent in sents.split("\1")]
            all_trees = [tree for tree in trees.split("\1")]

            tmp_model_inputs = self.tokenizer(all_sents,
                                              max_length=self.max_src_length,
                                              padding=False,
                                              truncation=True)
            all_tokenized_sents = [
                srci
                + [self.tokenizer.amr_bos_token_id,
                   self.tokenizer.mask_token_id,
                   self.tokenizer.amr_eos_token_id]
                for srci in tmp_model_inputs["input_ids"]
            ]

            all_tokenized_amrs = [
                [self.tokenizer.bos_token_id,
                 self.tokenizer.mask_token_id,
                 self.tokenizer.eos_token_id,
                 self.tokenizer.amr_bos_token_id]
                + self.tokenizer.tokenize_amr(tree.split())[:self.max_src_length - 1]
                + [self.tokenizer.amr_eos_token_id]
                for tree in all_trees
            ]

            model_inputs = {"input_ids": all_tokenized_sents,
                            "tree_ids": all_tokenized_amrs}

            return model_inputs

        self.test_dataset = datasets["test"].map(tokenize_function,
                                                 batched=False, remove_columns=["sent", "tree"], num_proc=8)
        print(f"ALL {len(self.test_dataset)} test instances")


@dataclass
class DataCollatorForEventTagging:
    """
    Squeeze the batch dimension.
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        features = features[0]

        return {
            "input_ids": features["input_ids"],
            "input_type_ids": features["input_type_ids"],
            "sent_ids": features["sent_ids"],
            "labels": features["labels"],
        }


@dataclass
class DataCollatorForEventGraph:
    """
    Squeeze the batch dimension.
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        features = features[0]

        return {"input_ids": features["input_ids"],
                "tree_ids": features["tree_ids"]}