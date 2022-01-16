#!/usr/bin/env python
# coding=utf-8
# Adapted from the Huggingface library
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

main_dir = os.path.abspath(os.path.dirname(__file__)).split('flexibert')[0]

from datasets import load_dataset, interleave_datasets
from transformers.models.bert.modeling_modular_bert import BertModelModular, BertForMaskedLMModular
from transformers import RobertaTokenizer, BertConfig

import transformers
import shlex
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
import argparse

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0.dev0")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )

    '''
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
    '''

def save_dataset(args):

    parser = HfArgumentParser(DataTrainingArguments)
    if len(args) == 2 and args[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args= parser.parse_json_file(json_file=os.path.abspath(args[1]))
    else:
        data_args= parser.parse_args_into_dataclasses(args)

    data_args = data_args[0]
    
    datasets = load_dataset('bookcorpus','plain_text',cache_dir='../')

    cc_news_valid = load_dataset('cc_news','plain_text',cache_dir='../',split=f"train[:{data_args.validation_split_percentage}%]")
    non_text_column_names = [name for name in cc_news_valid.column_names if name != 'text']
    cc_news_valid = cc_news_valid.remove_columns(non_text_column_names)

    bookcorpus_valid = load_dataset('bookcorpus','plain_text',cache_dir='../',split=f"train[:{data_args.validation_split_percentage}%]")
    #openwebtext = load_dataset('openwebtext','plain_text',cache_dir='../')
    wikipedia_valid = load_dataset('wikipedia','20200501.en',cache_dir='../',split=f"train[:{data_args.validation_split_percentage}%]")
    wikipedia_valid = wikipedia_valid.remove_columns('title')
    openwebtext_valid = load_dataset('openwebtext','plain_text',cache_dir='../',split=f"train[:{data_args.validation_split_percentage}%]")
    c4_valid = load_dataset('c4','en',cache_dir='../',split=f"validation")
    c4_valid = c4_valid.remove_columns('timestamp')
    c4_valid = c4_valid.remove_columns('url')
    combined_valid = interleave_datasets([cc_news_valid,bookcorpus_valid,wikipedia_valid,openwebtext_valid,c4_valid])

    datasets['validation'] = combined_valid

    cc_news_train = load_dataset('cc_news','plain_text',cache_dir='../',split=f"train[{data_args.validation_split_percentage}%:]")
    cc_news_train =  cc_news_train.remove_columns(non_text_column_names)
    bookcorpus_train = load_dataset('bookcorpus','plain_text',cache_dir='../',split=f"train[{data_args.validation_split_percentage}%:]")
    #openwebtext = load_dataset('openwebtext','plain_text',cache_dir='../')
    openwebtext_train = load_dataset('openwebtext','plain_text',cache_dir='../',split=f"train[{data_args.validation_split_percentage}%:]")
    wikipedia_train = load_dataset('wikipedia','20200501.en',cache_dir='../',split=f"train[{data_args.validation_split_percentage}%:]")
    wikipedia_train = wikipedia_train.remove_columns('title')
    c4_train = load_dataset('c4','en',cache_dir='../',split=f"train")
    c4_train = c4_train.remove_columns('timestamp')
    c4_train = c4_train.remove_columns('url')
    combined_train = interleave_datasets([cc_news_train,bookcorpus_train,wikipedia_train,openwebtext_train,c4_train])
    datasets['train'] = combined_train

    print("Dataset loaded")

    tokenizer = RobertaTokenizer.from_pretrained(main_dir+'roberta_tokenizer/')
    column_names = datasets["train"].column_names   
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )


    tokenized_datasets.save_to_disk(f'../tokenized_pretraining_dataset')


def get_args():

    a = "--max_seq_length 512"

    return shlex.split(a)

def main():


    parser = argparse.ArgumentParser(
        description='Input parameters for preparing datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()

    args_data = get_args()

    save_dataset(args_data)


if __name__ == '__main__':
    
    main()