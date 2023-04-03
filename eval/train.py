import logging
import os
os.environ['HF_HOME']="/scratch365/apoudel/.cache/huggingface"
os.environ['TRANSFORMERS_CACHE']="/scratch365/apoudel/.cache"
import re
import sys
import pandas as pd
from dataclasses import field, dataclass
from typing import Optional
import torch
from torch import nn
import numpy as np
import ast
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import transformers
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification
)
from transformers import EarlyStoppingCallback
from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint, is_main_process, set_seed
from eval.model import NLIPrediction
from eval.trainer import NLITrainer, compute_metrics_nli
from sklearn.metrics import average_precision_score 
import json
LABEL = 'label'
PREMISE = 'premise'
HYPOTHESIS = 'hypothesis'

pretrained_model="roberta-base"


logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        default=pretrained_model,
        metadata= {
            "help": 'name or path of the model from huggingface.co/models'
            }
    )

    config_name: Optional[str] = field(
        default = None, 
        metadata = {
            "help": "pretrained config name if not the same as model_name"
        }
    )

    tokenizer_name: Optional[str] = field(
        default= None, 
        metadata = {
            "help": "pretrained tokenizer if different from the model"
        }
    )

@dataclass
class DataTrainingArguments:

    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Cache dir for dataloader"
        }
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )

    save_preprocessed_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Save preprocessed dataset to disk for the usages in the future"
        },
    )
    preprocess_only: Optional[bool] = field(
        default=False, metadata={"help": "Return after finishing preprocessing"}
    )

    load_preprocessed_data: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether the data in given dir is already preprocessed"},
    )

    write_predict: bool = field(
        default=True,
        metadata={
            "help": "Write the preidction results under the output_dir if do_predict is set to true"
        },
    )
    predict_output_file: str = field(
        default="pred_output.csv",
        metadata={
            "help": "File name of the predict results. The file will be written under the output_dir"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

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
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )




def main(
    train_dataset=None,
    valid_dataset=None,
    test_dataset=None,
):


    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(
    #         json_file=os.path.abspath(os.system.argv[1])
    #     )
    # else:
    #     b_args = parser.parse_args_into_dataclasses(return_remaining_strings = True)[0]
    #     print(b_args)
    #     # exit()
    #     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    b_args = parser.parse_args_into_dataclasses(return_remaining_strings = True)

    model_args, data_args, training_args = b_args[0], b_args[1], b_args[2]
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.WARNING,
    )

    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Training/evaluation parameters {training_args}")
    last_checkpoint = None

    # training_args.output_dir = os.path.join(os.getcwd(), 'output')

    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    set_seed(training_args.seed)
    print(training_args.seed)
    # exit()


    #Configurations:

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    config.num_labels = 3
    print(config.num_labels)
    pad_on_right = tokenizer.padding_side == 'right'
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of = 8 if training_args.fp16 else None
        )
    )

    #preprocessinng the data -- tokenization
    def preprocess_function(example):

        result = tokenizer(
                    example['premise'],
                    example['hypothesis'],
                    padding = 'max_length',
                    max_length = max_seq_length,
                    truncation = True
                )
        return result


    #load the dataset

    raw_datasets = load_dataset('indonli')

    logger.info(raw_datasets)

    # train_datasaet = load_dataset('indonli', split = 'train[:1%]')

    # train_dataset = raw_datasets['train']
    eval_dataset = raw_datasets['validation']
    test_lay = raw_datasets['test_lay']
    test_expert = raw_datasets['test_expert']

    print(raw_datasets)

    model = NLIPrediction.from_pretrained(model_args.model_name_or_path, config = config)
    compute = compute_metrics_nli

    tokenized_datasets= raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        num_proc=data_args.preprocessing_num_workers,
        desc="Running tokenizer on dataset"
    )

    
    tokenized_datasets = tokenized_datasets.remove_columns(['premise', 'hypothesis'])
    print(tokenized_datasets)

    trainer = NLITrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset= train_dataset,
        eval_dataset= eval_dataset,
        args = training_args,
        data_collator = data_collator,
        compute_metrics = compute,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )


    if training_args.do_train:

        trainer.train_dataset = tokenized_datasets['train']
        trainer.eval_dataset = tokenized_datasets['validation']

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        print(100*"**")
        print('Training started!!')
        train_result  = trainer.train(resume_from_checkpoint = checkpoint)

        trainer.save_model()  ##Saves the tokenizer too
        metrics = train_result.metrics

        # max_train_samples = (
        #     data_args.max_train_samples
        #     if data_args.max_train_samples is not None
        #     else len(tokenized_datasets['train'])
        # )

        # print(max_train_samples)

        # metrics["train_samples"] = min(
        #     max_train_samples, len(tokenized_datasets['train'])
        # )


        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
    
    #Evaluation

    if training_args.do_eval:
        logger.info("Validation")
        metrics = trainer.evaluate(tokenized_datasets['validation'])

        # max_eval_samples = (
        #     data_args.max_eval_samples
        #     if data_args.max_eval_samples is not None
        #     else len(eval_dataset)
        # )
        # metrics["eval_samples"] = min(
        #     max_eval_samples, len(eval_dataset)
        # )

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #Prediction

    if training_args.do_predict:
        logger.info("*** Predict ***")
        results_dict = dict()
        for test_dataset in ['test_lay', 'test_expert']:
            results = trainer.predict(tokenized_datasets[test_dataset])
            metrics = results.metrics

            results_dict[test_dataset] = metrics

            pred_df = get_prediction_df(results.predictions, raw_datasets[test_dataset])
            pred_df = pred_df.sort_values("pred", ascending=False)

            if data_args.write_predict:
                pred_outfile = os.path.join(
                    training_args.output_dir, test_dataset + data_args.predict_output_file
                )
                pred_df.to_csv(
                    pred_outfile,
                    index=False,

                    columns=[
                        PREMISE,
                        HYPOTHESIS,
                        "pred",
                        LABEL,
                    ]
                )

                print(f"written to {pred_outfile}")

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

            print(f"Completed for {test_dataset}")
        
        with open(os.path.join(training_args.output_dir, "result_dict.json"), "w") as outfile:
            json.dump(results_dict, outfile) 



def get_prediction_df(preds, examples, append_text=False):
    
    print(preds)
    df = pd.DataFrame(columns={'premise', 'hypothesis', "pred", 'label'})
    # m = torch.nn.Softmax(dim=1)
    # print('preds', preds)
    pred_scores = torch.from_numpy(preds).float()[:, 2].tolist()

    assert(len(pred_scores) == len(examples))
    
    for i in range(len(examples)):

        df = df.append(
            {
                'premise': examples['premise'][i],
                'hypothesis': examples['hypothesis'][i],
                "pred": pred_scores[i],
                'label': examples['label'][i]
            },
            ignore_index=True,
        )

    return df


if __name__ == "__main__":

    main()
