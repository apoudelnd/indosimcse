import pandas as pd
import numpy as np
from transformers.utils import logging
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve,
)
from torch.utils.data.sampler import Sampler
import torch
from transformers.trainer import Trainer
import ast
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
from sklearn.metrics import confusion_matrix
import math


from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)

from transformers import (
    Trainer,
    EvalPrediction,
    is_torch_tpu_available,
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_torch_tpu_available,
    set_seed,
)
from datasets import Dataset

# from model.dataset import ModelsDataset
LINK = 'link_pred'
SATISFACTION = 'satisfaction'
MULTITASK = 'multitask'


logger = logging.get_logger(__name__)
logging.set_verbosity_warning()


class NLITrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset

        train_sampler = RandomSampler(train_dataset)

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler= train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        print(data_loader)

        return data_loader


def compute_metrics_nli(pred: EvalPrediction):

    logger.info("compute F-score metrics")
    labels = pred.label_ids.flatten()

    pred_labels = pred.predictions.argmax(-1).flatten()
    
    #average = 'binary' for binary classification!
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_labels, average='macro'
    )
    
    print(precision, recall, f1)

    acc = accuracy_score(labels, pred_labels)

    from sklearn.metrics import classification_report
    target_names = ['entailment', 'neutral', 'contradiction']
    print(labels)
    print(pred_labels)
    print(classification_report(labels, pred_labels, target_names=target_names))

    res = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    print(res)

    return res
