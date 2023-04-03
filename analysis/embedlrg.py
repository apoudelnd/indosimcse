import os
import numpy as np
import pandas as pd
import random
import math
from collections import defaultdict
import json
from numpy import loadtxt
import random
import faiss

import transformers

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from collections import defaultdict
# from sentence_transformers import SentenceTransformer
import random
random.seed(123)


tokenizer = AutoTokenizer.from_pretrained('indolem/indobertweet-base-uncased')
model = AutoModel.from_pretrained('indolem/indobertweet-base-uncased')
# model = SentenceTransformer('indolem/indobertweet-base-uncased')
# create sentence embeddings
# sentence_embeddings = model.encode(sentences)

data_folder = "/afs/crc.nd.edu/user/a/apoudel/projects/indosimcse/data/output_pd.csv"

def normalize_vec(arr, eps=1e-6):
    """ normalize a ndarray to (0,1] """
    return (arr - np.min(arr) + eps) / (np.max(arr) - np.min(arr) + eps)


b = np.load('/scratch365/apoudel/indosimcse/model/simcse-embed/output/2023_02_09_11_11_44_0/embedding.npy', allow_pickle=True)
df = pd.read_csv(data_folder)
df = df.dropna()
df['embeddings'] = list(b)

print(b.shape)
df = df.drop_duplicates(subset = ['text'])


print(np.array(list(df['embeddings'])).shape)

sentence_embeddings = np.array(list(df['embeddings']))
d = sentence_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
print(index.is_trained)


index.add(sentence_embeddings)
print(index.ntotal)

n_choices = 10
xq = random.sample(range(len(df)), k=n_choices)
print(xq)
print(xq[5])

print(df['text'].iloc[xq[5]])



with open('file.out', 'w') as fout:
    for i in xq:
        embed_xq = np.array(df['embeddings'].iloc[i])
        y = np.expand_dims(embed_xq, axis=0)
        k=4
        D, I = index.search(y, k)  # search
        print(I)
        fout.write(f"indexes {I} \n")
        for ii in I:
            for iii in ii:
                print(df['text'].iloc[iii])
                fout.write(f"{df['text'].iloc[iii]} \n")
        fout.write(100*"**"+ "\n")
        print(100*"**") 


