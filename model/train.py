import logging
import os
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
    AutoModelForSequenceClassification,
    AutoModel
)

import random
random.seed(123)

from tqdm import tqdm

print(torch.version.cuda)

import cuml
from cuml.manifold import TSNE, UMAP

pretrained_model = "indolem/indobertweet-base-uncased"
logger = logging.getLogger(__name__)

UID = "uid"
TEXT = "text"
VEC = "vec"

@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        default=pretrained_model,
        metadata={
            "help":'name or path of the model from huggingface.co/models'
        }
    )


@dataclass

class DataTrainingArguments:

    data_folder: str = field(
        default = "/afs/crc.nd.edu/user/a/apoudel/projects/indosimcse/data/output_pd.csv", 
        metadata = {
            "help": "path to the directory where the data resides"
        }
    )

    batch_size: int = field(
        default = 64,
        metadata = {
            "help": "batches of sentences to process -- mini-batches"
        }
    )

    image_dir: str = field(
        default = "/afs/crc.nd.edu/user/a/apoudel/projects/indosimcse/model/images", 
        metadata = {
            "help": "path to the directory where the data resides"
        }
    )

    max_seq_length: int = field(
        default = 256,
        metadata = {
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        }
    )

    reduction_teq: str = field(
        default = 'UMAP',
        metadata = {
            "help": "two options available - TSNE and UMAP"
        }
    )

    predict_output_file: str = field(
    default="outputvec.csv",
    metadata={
        "help": "File name of the predict results. The file will be written under the output_dir"
        },
    )

def normalize_vec(arr, eps=1e-6):
    """ normalize a ndarray to (0,1] """
    return (arr - np.min(arr) + eps) / (np.max(arr) - np.min(arr) + eps)

def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    b_args = parser.parse_args_into_dataclasses(return_remaining_strings = True)

    model_args, data_args, train_args = b_args[0], b_args[1], b_args[2]
  
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    pad_on_right = tokenizer.padding_side == 'right'
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    model = AutoModel.from_pretrained(model_args.model_name_or_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model.to(device)

    # def process_setences(dataset):

    df = pd.read_csv(data_args.data_folder)
    df = df.dropna()
    logger.info(df.head())
    
    logger.info(
        f"The size of the dataset is {df.shape}"
    )


    global_idx = 0
    vecs = list()

    def normalize_vec(arr, eps=1e-6):
        """ normalize a ndarray to (0,1] """
        return (arr - np.min(arr) + eps) / (np.max(arr) - np.min(arr) + eps)

    def tsne_redux(vec):
        """
        takes in a sth* 768 ndarray and reduces into * 2 dimensional vectors
        """
        # from sklearn.manifold import TSNE
        X_embedded = TSNE(n_components=2, perplexity=10).fit_transform(vec)
        return X_embedded

    def umap_redux(vec, n_neighbors):
        clus_emb = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            verbose=True,
        ).fit_transform(vec)

        # x_mean = np.mean(clus_emb[:, 0])
        # y_mean = np.mean(clus_emb[:, 1])
        # x_std = np.std(clus_emb[:, 0])
        # y_std = np.std(clus_emb[:, 1])

        # clus_emb = clus_emb[clus_emb[:, 0] < x_mean + 3 * x_std]
        # clus_emb = clus_emb[clus_emb[:, 0] > x_mean - 3 * x_std]
        # clus_emb = clus_emb[clus_emb[:, 1] < y_mean + 3 * y_std]
        # clus_emb = clus_emb[clus_emb[:, 1] > y_mean - 3 * y_std]

        return clus_emb

    def write_to_csv(embeddings):

        """
        takes in a 2d ndarray and writes to a csv file
        """
        df[VEC] = list(embeddings)
        pred_outfile = os.path.join(
                train_args.output_dir, data_args.predict_output_file
                )
        df.to_csv(pred_outfile)

    def plotembeddings(X, Y, n):

        import matplotlib.pyplot as plt

        fig = plt.figure(1, figsize=(100, 80), dpi=60)
        plt.scatter(X, Y, s=2) #change it later , s=2
        plt.savefig(os.path.join(data_args.image_dir, f'graph_{data_args.reduction_teq}_{n}.png'))

    batch_size = data_args.batch_size

    for idx in tqdm(range(0, len(df[UID]), batch_size)):
        batch = list(df[TEXT])[idx: min(len(df[UID]), idx+batch_size)]

        encoded = tokenizer.batch_encode_plus(batch,
                                            max_length = max_seq_length,
                                            padding = 'max_length', 
                                            truncation = True)

        encoded = {key:(torch.LongTensor(value)).to(device) for key, value in encoded.items()}

        with torch.no_grad():
        
            outputs = model(**encoded)

        last_hidden = outputs.last_hidden_state[:,0].detach().cpu().numpy()
        vecs.append(last_hidden)

    x = np.concatenate(vecs)
    # print(x)
    # ids = np.asarray(list(df["uid"]))
    # print(ids)

    # assert(len(ids)==len(x))
    # exit()
    # from numpy import savetxt
    # savetxt(os.path.join(train_args.output_dir, 'embedding.csv'), x, delimiter=',')

    np.save(os.path.join(train_args.output_dir, 'embedding.npy'), np.array(x, dtype=object), allow_pickle=True)

    logger.info(f"size of sentence embeddings: {x.shape}")

    """different number of neighbors"""
    for n in [5, 10, 15, 20]:
        embeddings = umap_redux(x, n) if data_args.reduction_teq == 'UMAP' else tsne_redux(x)

        logger.info(f"Completed Dimensionality reduction for {n} neighbors")

        embeddings[:,0] = normalize_vec(embeddings[:,0])
        embeddings[:,1] = normalize_vec(embeddings[:,1])

        write_to_csv(embeddings)

        plotembeddings(embeddings[:, 0], embeddings[:, 1], n)

        logger.info(f"plotted for {n} neighbors")


        break

        
if __name__ == "__main__":

    b = np.load('/scratch365/apoudel/indosimcse/model/simcse-embed/output/2023_02_09_11_11_44_0/embedding.npy', allow_pickle=True)

    clus_emb = UMAP(
            n_components=2,
            n_neighbors=15,
            verbose=True,
        ).fit_transform(b)

    x_mean = np.mean(clus_emb[:, 0])
    y_mean = np.mean(clus_emb[:, 1])
    x_std = np.std(clus_emb[:, 0])
    y_std = np.std(clus_emb[:, 1])

    def normalize_vec(arr, eps=1e-6):
        """ normalize a ndarray to (0,1] """
        return (arr - np.min(arr) + eps) / (np.max(arr) - np.min(arr) + eps)

    clus_emb = clus_emb[clus_emb[:, 0] < x_mean + 3 * x_std]
    clus_emb = clus_emb[clus_emb[:, 0] > x_mean - 3 * x_std]
    clus_emb = clus_emb[clus_emb[:, 1] < y_mean + 3 * y_std]
    clus_emb = clus_emb[clus_emb[:, 1] > y_mean - 3 * y_std]

    logger.info("Completed Dimensionality reduction")

    clus_emb[:,0] = normalize_vec(clus_emb[:,0])
    clus_emb[:,1] = normalize_vec(clus_emb[:,1])

    # write_to_csv(embeddings)

    import matplotlib.pyplot as plt

    fig = plt.figure(1, figsize=(100, 80), dpi=60)
    plt.scatter(clus_emb[:, 0], clus_emb[:, 1], s=2) #change it later , s=2
    plt.savefig('graph_withoutlier.png')

    # plotembeddings(embeddings[:, 0], embeddings[:, 1])

    logger.info("plotted")


    # main()



