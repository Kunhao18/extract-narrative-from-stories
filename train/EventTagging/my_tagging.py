### THIS FILE IS COPIED FROM THE HUGGINGFACE REPOSITORY FOR CONVENIENCE.

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import argparse
import glob
import logging
import os
import json
from posixpath import join
import random
import re
import shutil
from typing import Dict, List, Any, Tuple
from data_interface.dataset import EventTaggingDataSet, DataCollatorForEventTagging
from model_interface.modeling_bart import BartForConditionalGeneration, BartForEventTagging
from model_interface.tokenization_bart import AMRBartTokenizer

from transformers import BertTokenizer, BertModel

from torch.cuda.amp import autocast as autocast

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from common.utils import (
    get_STD2partial,
    get_MTEG2text,
    get_ETMG2graph,
    get_PTPG2partial,
    get_MTMG2partial,
    get_MTMG2TG,
    get_inverse_sqrt_schedule_with_warmup,
    save_dummy_batch,
)

from sklearn.metrics import confusion_matrix, accuracy_score

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# from transformers.modeling_auto import MODEL_WITH_LM_HEAD_MAPPING

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            "Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint)
        )
        shutil.rmtree(checkpoint)


def train(
        args,
        train_dataset,
        eval_dataset,
        collate_fn,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config,
        story_model=None,
) -> Tuple[int, float, str]:
    """ Train the model """
    tb_writer = SummaryWriter()

    train_sampler = (RandomSampler(train_dataset))
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  collate_fn=collate_fn)
    # TODO: Doesn't support batch now

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1)
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_inverse_sqrt_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=args.warmup_steps,
                                                      num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if (args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num epochs = %d", args.num_train_epochs)
    logger.info("  Actual batch size = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    epoch_step = 0
    steps_trained_in_current_epoch = 0
    best_score = -1
    best_out_path = None
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss, epoch_loss = 0.0, 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(epochs_trained,
                            int(args.num_train_epochs),
                            desc="Epoch")
    set_seed(args)  # Added here for reproducibility
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            with autocast():
                # Story embedding
                story_embed = None
                if story_model:
                    with torch.no_grad():
                        story_input = batch["input_ids"]
                        story_type_input = batch["input_type_ids"]
                        story_input = torch.LongTensor([story_input]).to(args.device)
                        story_type_input = torch.LongTensor([story_type_input]).to(args.device)
                        story_output = story_model(story_input, token_type_ids=story_type_input,
                                                   return_dict=True)
                        story_embed = story_output["last_hidden_state"][:, 0, :]

                # Tagging task
                sent_input = batch["sent_ids"]
                labels = batch["labels"]
                labels = torch.LongTensor(labels).to(args.device)
                outputs = model(
                    input_ids=sent_input,
                    labels=labels,
                    story_embeddings=story_embed
                )
                loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            epoch_iterator.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

            loss.backward()
            epoch_step += 1
            tr_loss += loss.item()
            epoch_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        results = evaluate(args, eval_dataset, collate_fn, model, tokenizer,
                                           config=config, story_model=story_model)
                        cur_score = results["eval_f2"]

                        if cur_score > best_score:
                            best_score = cur_score
                            checkpoint_prefix = "checkpoint"
                            # Save model checkpoint
                            output_dir = os.path.join(args.output_dir,
                                                      "{}-{}-{:.3f}".format(checkpoint_prefix, global_step, best_score))
                            best_out_path = output_dir
                            os.makedirs(output_dir, exist_ok=True)
                            model.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            _rotate_checkpoints(args, checkpoint_prefix)

                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            results = evaluate(args, eval_dataset, collate_fn, model, tokenizer, config=config,
                               story_model=story_model)
            cur_score = results["eval_f2"]
            checkpoint_prefix = "checkpoint"
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "{}-last-{:.3f}".format(checkpoint_prefix, cur_score))
            if cur_score > best_score:
                best_score = cur_score
                best_out_path = output_dir
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)
            train_iterator.close()
            break

        results = evaluate(args, eval_dataset, collate_fn, model, tokenizer, config=config,
                           story_model=story_model)
        cur_score = results["eval_f2"]
        if cur_score > best_score:
            best_score = cur_score
            checkpoint_prefix = "checkpoint"
            output_dir = os.path.join(args.output_dir, "{}-{}-{:.3f}".format(checkpoint_prefix,
                                                                             global_step,
                                                                             best_score))
            best_out_path = output_dir
            os.makedirs(output_dir, exist_ok=True)

            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            _rotate_checkpoints(args, checkpoint_prefix)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)

            avg_epoch_loss = epoch_loss / epoch_step
            logger.info("\nEpoch End... \navg_train_loss = %s", str(avg_epoch_loss))

    tb_writer.close()

    return global_step, tr_loss / global_step, best_out_path


def evaluate(
        args,
        eval_dataset,
        collate_fn,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config=None,
        story_model=None,
        test_output=None
) -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    os.makedirs(eval_output_dir, exist_ok=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, collate_fn=collate_fn)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    pbar = tqdm(eval_dataloader, desc="Evaluating")
    output_pred = []
    output_true = []
    output_result = []

    for batch in pbar:
        with torch.no_grad():
            story_embed = None
            if story_model:
                story_input = batch["input_ids"]
                story_type_input = batch["input_type_ids"]
                story_input = torch.LongTensor([story_input]).to(args.device)
                story_type_input = torch.LongTensor([story_type_input]).to(args.device)
                story_output = story_model(story_input, token_type_ids=story_type_input,
                                           return_dict=True)
                story_embed = story_output["last_hidden_state"][:, 0, :]

            sent_input = batch["sent_ids"]
            labels = batch["labels"]
            labels = torch.LongTensor(labels).to(args.device)

            outputs = model(input_ids=sent_input,
                            labels=labels,
                            story_embeddings=story_embed)
            loss = outputs[0]

            tag_logits = outputs[1][0]
            output_pred.extend(torch.argmax(tag_logits, dim=1).tolist())
            output_result.append(torch.argmax(tag_logits, dim=1).tolist())
            output_true.extend(labels.tolist())

            pbar.set_postfix(loss=loss.mean().item())
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

    if test_output:
        with open(test_output, "w") as f_out:
            json.dump(output_result, f_out)

    eval_loss = eval_loss / nb_eval_steps

    tn, fp, fn, tp = confusion_matrix(output_true, output_pred).ravel()
    eval_acc = accuracy_score(output_true, output_pred)
    eval_rec = tp / (tp + fn) if tp != 0 else 0.
    eval_pre = tp / (tp + fp) if tp != 0 else 0.
    eval_f1 = 2 * eval_rec * eval_pre / (eval_pre + eval_rec) if tp != 0 else 0.
    eval_f2 = 5 * eval_rec * eval_pre / (4 * eval_pre + eval_rec) if tp != 0 else 0.

    result = {"eval_loss": eval_loss,
              "eval_acc": eval_acc,
              "eval_pre": eval_pre,
              "eval_rec": eval_rec,
              "eval_f1": eval_f1,
              "eval_f2": eval_f2}

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("\n***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--val_file", default=None, type=str, required=True,
                        help="The input validation data file (a text file).")
    parser.add_argument("--test_file", default=None, type=str, required=True,
                        help="The input testing data file (a text file).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Dir & model parameters
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization. "
                             "Leave None if you want to train a model from scratch.")
    parser.add_argument("--story_model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization. "
                             "Leave None if you want to train a model from scratch.")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path. "
                             "If both are None, initialize a new config.")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3"
                             " (instead of the default one)")

    # Train & eval parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--should_continue", action="store_true",
                        help="Whether to continue from latest checkpoint in output_dir")

    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=None,
                        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, "
                             "does not delete by default")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs "
                             "(take into account special tokens).")

    # Optimizer parameters
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # Other parameters
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
            and not args.should_continue):
        raise ValueError("Output directory ({}) already exists and is not empty. "
                         "Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError("You are instantiating a new config instance from scratch. "
                         "This is not supported, but you can do it from another script, save it,"
                         "and load it from here, using --config_name.")

    tokenizer = AMRBartTokenizer.from_pretrained(args.model_name_or_path)

    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)

    # Load model
    if args.model_name_or_path:
        model = BartForEventTagging.from_pretrained(args.model_name_or_path,
                                                    from_tf=bool(".ckpt" in args.model_name_or_path),
                                                    config=config,
                                                    cache_dir=args.cache_dir)
    else:
        logger.info("Training new model from scratch")
        model = BartForEventTagging.from_config(config)

    model.to(args.device)
    # print(model)
    # train_p = [n for n, p in model.named_parameters() if p.requires_grad]
    # print(f"Trainable params in Summarization Model : {train_p}")

    # Load story model (BERT)
    story_tokenizer = BertTokenizer.from_pretrained(args.story_model_name_or_path) \
        if args.story_model_name_or_path else None
    story_model = BertModel.from_pretrained(args.story_model_name_or_path) \
        if args.story_model_name_or_path else None
    if story_model:
        story_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    event_dataset = EventTaggingDataSet(tokenizer=tokenizer,
                                        story_tokenizer=story_tokenizer,
                                        train_file=args.train_file,
                                        validation_file=args.val_file,
                                        test_file=args.test_file,
                                        pad_to_max_length=False,
                                        max_src_length=args.block_size,
                                        max_tgt_length=256)
    event_dataset.setup()

    # Dummy Test
    train_dataset = event_dataset.train_dataset
    dev_dataset = event_dataset.valid_dataset
    test_dataset = event_dataset.test_dataset

    event_tagging_collate_fn = DataCollatorForEventTagging(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
    )

    # Training
    if args.do_train:
        global_step, tr_loss, best_path = train(args, train_dataset, dev_dataset, event_tagging_collate_fn,
                                                model, tokenizer, config, story_model=story_model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Load the model with the best eval results
        model = BartForEventTagging.from_pretrained(best_path)
        tokenizer = AMRBartTokenizer.from_pretrained(best_path)
        model.to(args.device)

        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info("Saving best model checkpoint to %s", args.output_dir)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    result = {}
    if args.do_test:
        checkpoint = args.output_dir
        logger.info("Evaluate the following checkpoints: %s", checkpoint)

        model = BartForEventTagging.from_pretrained(checkpoint)
        model.to(args.device)

        story_model = BertModel.from_pretrained(args.story_model_name_or_path) \
            if args.story_model_name_or_path else None
        if story_model:
            story_model.to(args.device)

        test_out_path = os.path.join(args.output_dir, "test_result.json")
        result = evaluate(args, test_dataset, event_tagging_collate_fn, model, tokenizer,
                          config=config, story_model=story_model, test_output=test_out_path)

    return result


if __name__ == "__main__":
    main()
