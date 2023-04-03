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

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

# tokenizer = AutoTokenizer.from_pretrained('/scratch365/apoudel/indosimcse/SimCSE/output/2023_01_20_12_41_05_12/checkpoint-4500')
# model = AutoModel.from_pretrained('/scratch365/apoudel/indosimcse/SimCSE/output/2023_01_20_12_41_05_12/checkpoint-4500')

# sentences = [
#     "Three years later, the coffin was still full of Jello.",
#     "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
#     "The person box was packed with jelly many dozens of months later.",
#     "He found a leprechaun in his walnut shell."
# ] --  first sentence compared with others..
# [[0.3201306  0.60089695 0.35318708]]

tokenizer = AutoTokenizer.from_pretrained('indolem/indobertweet-base-uncased')
model = AutoModel.from_pretrained('indolem/indobertweet-base-uncased')
# [[0.7015468  0.84120464 0.6234921 ]]

# batch = ["kementerian komunikasi dan informatika",
#             "layanan internet di tanah air berkembang pesat",
#             "china diprediksi pulih dari pandemi pada pertengahan 2023",
#             "guru membantu anakanak dengan tugas sekolah foto stok - unduh gambar  sekarang - aktivitas bergerak, anak - umur manusia, anak laki-laki -  laki-laki - istock",
#             "inflasi jepang tembus 3,7 persen, tertinggi dalam 40 tahun terakhir",
#             "tbc mewabah, di kabupaten tegal dalam setahun ditemukan 3.858 kasus, 669  dianataranya anak-anak"]

# sentences = [
#     "Three years later, the coffin was still full of Jello.",
#     "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
#     "The person box was packed with jelly many dozens of months later.",
#     "He found a leprechaun in his walnut shell."
# ]

batch = ["Tiga tahun kemudian, peti mati itu masih penuh dengan Jello.",
     "Ikan bermimpi melarikan diri dari akuarium dan masuk ke toilet di mana dia melihat temannya pergi.",
     "Kotak orang itu penuh dengan jeli puluhan bulan kemudian.",
     "Dia menemukan leprechaun di kulit kenarinya."]

batch_x = ["kejar capaian vaksin covid-19, pemdes sranak bersama forpimcam trucuk plus  gelar gebyar vaksin - tajuk online",
    "cerita kepsek mim patikraja terapkan ekonomi sirkular di sekolah, hemat  biaya-minim sampah",
    "infopublik - jelang pemilu 2024, badan kesbangpol buleleng-bali gaungkan  wawasan kebangsaan",
    "misa malam natal gereja santo fransiskus assisi makassar berbalut adat jawa"]


batch_y = ["Rusia adalah bangsa terbesar",
"Putin adalah pemimpin dunia",
"ada banyak informasi yang salah di web akhir-akhir ini",
"bumi adalah tempat yang indah untuk ditinggali",
"NLP adalah teknik paling populer dalam AI dan Pembelajaran Mesin saat ini",
"Film itu benar-benar garba"]   
    


max_seq_length = 64

encoded = tokenizer.batch_encode_plus(batch_y,
                                    max_length = max_seq_length,
                                    padding = 'max_length', 
                                    truncation = True)

encoded = {key:(torch.LongTensor(value)) for key, value in encoded.items()}

vecs = list()

with torch.no_grad():

    outputs = model(**encoded)

    last_hidden = outputs.last_hidden_state[:,0].detach().cpu().numpy()
    vecs.append(last_hidden)
        # for vec in last_hidden[:,0]:
        #     vecs.append(vec)
x = np.concatenate(vecs)

print(x)
print(x.shape)

print(cosine_similarity(
    [x[0]],
    x[1:]
))

dist_out = 1-pairwise_distances(x, metric="cosine")

print(dist_out)
