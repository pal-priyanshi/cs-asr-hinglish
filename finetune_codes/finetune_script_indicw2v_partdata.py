#!/usr/bin/env python
# coding=utf-8

""" Fine-tuning a 🤗 Transformers CTC model for automatic speech recognition"""

"""
Nhan documentation on the finetune script

The model might be different with other Finnish wav2vec2 model due to the following:

There's no <s> and </s> in the vocab.json, but they're in the special_tokens_map.json

There's difference when you call:
# Load vocab file (no added token)
with open(path_model + '/vocab.json', encoding='utf-8') as json_file:
    vocab = json.load(json_file)
    
or using (have added tokens):
vocab_dict = processor.tokenizer.get_vocab()  

This would cause some small problem when the logits matrix didn't output the added token in prediction
(since we never added <s> and </s> to the model)
For the purpose of mispronunciation problem, this won't cause any trouble

Newer version of the model was fixed when I did a new finetune, but nevertheless the old version has this problem
"""

import functools
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from transformers import Wav2Vec2ProcessorWithLM

import wandb
import datasets
import numpy as np
import torch
from datasets import DatasetDict, load_dataset, load_metric
from evaluate import load
import torch.nn as nn

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    set_seed,
    Wav2Vec2ForCTC,
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

#sys.path.insert(0,"/scratch/work/palp3/MUCS/dataset") # (import up one level of directory)
#from utils import custom_models as custom_models

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.16.0.dev0")

#require_version("datasets>=1.13.3", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = logging.getLogger(__name__)
#logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO)
logger.info("Is cuda available:" + str(torch.cuda.is_available()))
#logger.info("Device name: " + str(torch.cuda.get_device_name(0)))

def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)

MODEL_NAME = "ai4bharat/indicwav2vec-hindi"
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    attention_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    feat_proj_dropout: float = field(default=0.0, metadata={"help": "The dropout ratio for the projected features."})
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": "Probability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis."
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": "Probability of each feature vector along the feature axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature bins will be masked along the time axis."
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    ctc_loss_reduction: Optional[str] = field(
        default="mean", metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."},
    )

    ctc_zero_infinity: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to zero infinite losses and the associated gradients of ``torch.nn.CTCLoss``. Infinite losses mainly occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance of the transformers.Wav2Vec2ForCTC "},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    #output_dir: str = field(
    #    default=None,
    #   metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    #)
    dataset_name: str = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_path: str = field(
        default=None,
        metadata={"help": "The path to the csv file contain train set."}
    )
    eval_path: str = field(
        default=None,
        metadata={"help": "The path to the csv file contain evaluation set."}
    )
    dataset_config_name: str = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    train_split_name: str = field(
        default="train+validation",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    #group_by_length: Optional[bool] = field(
     #   default=False,
      #  metadata={"help": "Whether or not to group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient). Only useful if applying dynamic padding."},
    #)
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    train_sampling_rate: Optional[int] = field(
        default=16000,
        metadata={
            "help": "Set the target sampling rate incase of downsampling but still want to use 16kHz pretrained model"
        },
    )        
    chars_to_ignore: Optional[List[str]] = list_field(
        default=None,
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    eval_metrics: List[str] = list_field(
        default=["wer"],
        metadata={"help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"},
    )
    max_duration_in_seconds: float = field(
        default=40.0,
        metadata={
            "help": "Filter audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only do data preprocessing and skip training. "
            "This is especially useful when data preprocessing errors out in distributed training due to timeout. "
            "In this case, one should run the preprocessing in a non-distributed setup with `preprocessing_only=True` "
            "so that the cached datasets can consequently be loaded in distributed training"
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "If :obj:`True`, will use the token generated when running"
            ":obj:`transformers-cli login` as HTTP bearer authorization for remote files."
        },
    )
    unk_token: str = field(
        default="[UNK]",
        metadata={"help": "The unk token for the tokenizer"},
    )
    pad_token: str = field(
        default="[PAD]",
        metadata={"help": "The padding token for the tokenizer"},
    )
    word_delimiter_token: str = field(
        default="|",
        metadata={"help": "The word delimiter token for the tokenizer"},
    )
    phoneme_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "The target language that should be used be"
            " passed to the tokenizer for tokenization. Note that"
            " this is only relevant if the model classifies the"
            " input audio to a sequence of phoneme sequences."
        },
    )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    wandb.login()
    #wgas1fp16truebs8
    wandb.init(project="huggingface", name="rerun_bestrun_wgas1fp16false_indicw2v_ad0_3_hd_02_featd_0_3_lr6e-4_warmup500_s300_shuff100",group="rerun_best_models")
    #%env WANDB_PROJECT=your-project-name
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    #Detecting last checkpoint

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
        #handlers=[logging.FileHandler(filename="logs/log_file.txt", encoding='utf-8', mode='w+')],
    )

    # Log on each process the small summary:
    logger.warning(
        f"device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    #if is_main_process(training_args.local_rank):
     #   transformers.utils.logging.set_verbosity_info()
    #logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 1. First, let's load the dataset
    raw_datasets = DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_dataset(
            "json",
            data_files=data_args.train_path,
            field="train",
            cache_dir=model_args.cache_dir,
            #load_dataset("json", data_files="MUCS_train_test_dataset_dict.json", field="train", cache_dir= "/scratch/work/palp3/MUCS/dataset/")
        )["train"]

        if data_args.audio_column_name not in raw_datasets["train"].column_names:
            print(raw_datasets["train"].column_names, data_args.audio_column_name, raw_datasets["train"].values(), type(raw_datasets["train"]) )
            raise ValueError(
                f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--audio_column_name` to the correct audio column - one of "
                f"{', '.join(raw_datasets['train'].column_names)}."
            )
        #selecting only 20000 points for now.
        raw_datasets["train"] = raw_datasets["train"].shuffle(seed=100).select(range(20000))



        #if data_args.max_train_samples is not None:
         #   raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    if training_args.do_eval:
        raw_datasets["eval"] = load_dataset(
            "json",
            data_files=data_args.eval_path,
            field="test",
            cache_dir=model_args.cache_dir,
        )["train"]

        #if data_args.max_eval_samples is not None:
         #   raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

    # save special tokens for tokenizer
    word_delimiter_token = data_args.word_delimiter_token
    unk_token = data_args.unk_token
    pad_token = data_args.pad_token

    # 3. Next, let's load the config as we might need it to create
    # the tokenizer
    # load config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_auth_token=data_args.use_auth_token
    )
    # 5. Now we can instantiate the feature extractor, tokenizer and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.
    tokenizer = Wav2Vec2CTCTokenizer("/scratch/elec/puhe/p/palp3/MUCS/vocab1_MUCS.json", 
                                     unk_token=unk_token, 
                                     pad_token=pad_token, 
                                     word_delimiter_token=word_delimiter_token)
    
    print(tokenizer)
    # load feature_extractor and tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(
    #    tokenizer_name_or_path,
    #    use_auth_token=data_args.use_auth_token,
    #    **tokenizer_kwargs,
    #)
    
    #return_attention_mask=data_args.return_attention_mask
    #print("return_attention_mask: ")
    #print(data_args.return_attention_mask)
    #Wav2Vec2 models that have set config.feat_extract_norm == "group", such as wav2vec2-base, have not been trained using attention_mask. 
    # For such models, input_values should simply be padded with 0 and no attention_mask should be passed.
    # For Wav2Vec2 models that have set config.feat_extract_norm == "layer", such as wav2vec2-lv60, attention_mask should be passed for batched inference.
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_auth_token=data_args.use_auth_token
    )

    #feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

#feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)


    # adapt config
    config.update(
        {            
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": model_args.activation_dropout,
        }
    )

    # create model
    feature_extractor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    #processor.save_pretrained(training_args.output_dir)
    config.save_pretrained(training_args.output_dir)
    #processor = Wav2Vec2ProcessorWithLM(
    #feature_extractor=feature_extractor,
    #tokenizer=tokenizer,
    #)
    state_dict = torch.load(f"{model_args.model_name_or_path}/pytorch_model.bin")
    #state_dict = torch.load(f"/m/triton/scratch/elec/puhe/p/palp3/MUCS/indicwav2vec-hindi/pytorch_model.bin")
    state_dict.pop("lm_head.weight")
    state_dict.pop("lm_head.bias")
    #config = Wav2Vec2Config.from_pretrained(model_path)
    #processor = Wav2Vec2Processor.from_pretrained(model_path)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(training_args.output_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_args.model_name_or_path, state_dict=state_dict, config=config, cache_dir=model_args.model_name_or_path)
    print("CHECK MODEL PARAMS", model)
    model.save_pretrained(training_args.output_dir)


    #processor = AutoProcessor.from_pretrained(training_args.output_dir)
    #model = AutoModelForCTC.from_pretrained(
    #    pretrained_model_name_or_path= model_args.model_name_or_path,
    #    cache_dir=model_args.cache_dir,
    #    #ctc_loss_reduction="mean",
    #    config=config,

        #pad_token_id=processor.tokenizer.pad_token_id,
        #vocab_size=len(processor.tokenizer),
    #    ignore_mismatched_sizes=True,
    #    use_auth_token=data_args.use_auth_token,
    #    )
    #model.lm_head = nn.Linear(model.config.hidden_size, len(tokenizer))
    
    '''model = AutoModelForCTC.from_pretrained(
    "ai4bharat/indicwav2vec-hindi", 
    cache_dir="/scratch/work/palp3/MUCS/dataset/finetuning_files",
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True'''
    
#)
#processor = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec-hindi")

    # freeze encoder
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # 6. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`

    # make sure that dataset decodes audio with correct sampling rate
    #dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    #dataset_sampling_rate = 16000
    #if dataset_sampling_rate != feature_extractor.sampling_rate:
    #   raw_datasets = raw_datasets.cast_column(
    #       data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    #   )

    # Cast Audio to the file path "audio_column_name" so we can load our own dataset
    # And using Audio(sampling=...) to set target sampling rate
    #raw_datasets = raw_datasets.cast_column(data_args.audio_column_name, datasets.features.Audio(sampling_rate=data_args.train_sampling_rate))
    #train_waud = raw_datasets["train"]["train"].cast_column(data_args.audio_column_name,  datasets.Audio(sampling_rate=16000))
    #test_waud = raw_datasets["eval"]["train"].cast_column(data_args.audio_column_name,  datasets.Audio(sampling_rate=16000))
    raw_datasets = raw_datasets.cast_column(data_args.audio_column_name, datasets.Audio(sampling_rate=16000))


    # derive max & min input length for sample rate & max duration
    #max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    #min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers



    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio        
        sample = batch[data_args.audio_column_name]
        #inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        #inputs = feature_extractor(sample["array"], sampling_rate=16000)
        batch["input_values"] = processor(sample["array"], sampling_rate=16000).input_values[0]
        #batch["input_values"] = inputs.input_values[0]
        #batch["input_values"] = processor(batch[data_args.audio_column_name], sampling_rate=16000).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        # encode targets
        #additional_kwargs = {}
        
        batch["labels"] = tokenizer(batch[data_args.text_column_name]).input_ids
        return batch
    

    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=num_workers,
            cache_file_names={"train":"/scratch/elec/puhe/p/palp3/MUCS/train_dataset_cache_20000samples_seed300shufseed100.arrow", "eval":"/scratch/elec/puhe/p/palp3/MUCS/test_dataset_cache_v2.arrow"},
            desc="preprocess datasets",
        )
        #print("Testing file matches audio in test", test_waud[0])
        #print("Testing file matches audio in train", train_waud[0])

        '''
        train_waud = train_waud.map(
                    prepare_dataset,
                    cache_file_name="/scratch/elec/puhe/p/palp3/MUCS/train_dataset_cache.arrow",
                    num_proc=num_workers,
                    desc="preprocess train data",
                )
        test_waud = test_waud.map(
                    prepare_dataset,
                    cache_file_name="/scratch/elec/puhe/p/palp3/MUCS/test_dataset_cache.arrow",
                    num_proc=num_workers,
                    desc= "preprocess test data"
        )'''

        #def is_audio_in_length_range(length):
         #   return length > min_input_length and length < max_input_length

        # filter data that is shorter than min_input_length
        #vectorized_datasets = vectorized_datasets.filter(
        #    is_audio_in_length_range,
        #    num_proc=num_workers,
        #    input_columns=["input_length"],
        #)

    # 7. Next, we can prepare the training.
    # Let's use word error rate (WER) as our evaluation metric,
    # instantiate a data collator and the trainer

    # Define evaluation metrics during training, *i.e.* word error rate, character error rate
    eval_metrics = {metric: load(metric) for metric in data_args.eval_metrics}

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {cache_file_name}")
        return
    
    def save_predictions_to_file(predictions, references, filename):
        with open(filename, "a") as f:
            for pred, ref in zip(predictions, references):
                f.write(f"References: /{ref}\nPrediction: {pred}\n\n")

    def compute_metrics(pred, output_file="/scratch/elec/puhe/p/palp3/MUCS/indicwav2vec_outputs/pd_warmup500_rerun_latest/gas1fp16false_predictions_indicw2v_ad0_3_hd_02_featd_0_2_lr6e-4_warmup500_s300_shuff100.txt"):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
        pred_str_fw = [tokenizer.decode(ids) for ids in pred_ids]
        label_str_fw = [tokenizer.decode(ids) for ids in pred.label_ids]
        save_predictions_to_file(pred_str_fw, label_str_fw, output_file)
        #print("Printing logits")
        #print("\n")
        #print(pred_logits)
        print("Printing predictions for a few samples:")
        for i in range(min(5, len(pred_str))):
            print(f"Sample {i+1}:")
            print(f"  Reference: {label_str[i]}")
            print("######")
            print("\n")
            print(f"  Prediction: {pred_str[i]}")
            print("\n\n")
        print("last Reference string", label_str[-1])
        print("\n")
        print("last prediction string", pred_str[-1])
        #print("reference transcript=", label_str)
        #print("Predictions transcript=", pred_str)

        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}

        return metrics


    

    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    
    #optimizer = bnb.optim.Adam8bit(
    #    params=optimizer_grouped_parameters,
    #    lr=training_args.learning_rate,
    #    betas=(training_args.adam_beta1, training_args.adam_beta2),
    #    eps=training_args.adam_epsilon,
    #)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        tokenizer=feature_extractor,
        #optimizers=optimizers,
    )

    # 8. Finally, we can start training

    # Training
    if training_args.do_train:

        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        #train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    #config_name = data_args.dataset_config_name if data_args.dataset_config_name is not None else "na"
    #kwargs = {
    #    "finetuned_from": model_args.model_name_or_path,
    #    "tasks": "speech-recognition",
    #    "tags": ["automatic-speech-recognition", data_args.dataset_name],
    #    "dataset_args": f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split: {data_args.eval_split_name}",
    #    "dataset": f"{data_args.dataset_name.upper()} - {config_name.upper()}",
    #}
    #if "common_voice" in data_args.dataset_name:
    #    kwargs["language"] = config_name

    if training_args.push_to_hub:
        trainer.push_to_hub()
    else:
        trainer.create_model_card()

    return results


if __name__ == "__main__":
    main()