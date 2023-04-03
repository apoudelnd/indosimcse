import os
import sys
from PIL import Image, ImageFile
from transformers import CLIPProcessor

import pandas as pd
import torch
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import transformers
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

from dataclasses import field, dataclass
from dataset import CLIPDataset, create_data_loader

# import torchvision.transforms as transforms

import random
random.seed(123)

from tqdm import tqdm

print(torch.version.cuda)
#we need image_encoder, and text_encoder


img_pretrained_model = 'clip-ViT-B-32'
text_pretrained_model = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'

@dataclass

class ModelArguments:

    image_model_path: str = field(
        default = img_pretrained_model,
        metadata = {
            "help": 'name or path of the pretrained image encoder model'
        }
    )

    text_model_path: str = field(
        default = text_pretrained_model, 
        metadata = {
            "help": 'name or path of the pretrained text model'
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


def main():

    parser = HfArgumentParser(
        {ModelArguments, DataTrainingArguments, TrainingArguments}
    )

    b_args = parser.parse_args_into_dataclasses(return_remaining_strings = True)

    model_args, data_args, train_args = b_args[0], b_args[1], b_args[2]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    img_model = SentenceTransformer('clip-ViT-B-32')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = create_data_loader(df, batch_size = data_args.batch_size)

    def load_image(url_or_path):
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            return Image.open(requests.get(url_or_path, stream=True).raw)
        else:
            return Image.open(url_or_path)


    for batch in data_loader:

        uids = batch['uid']
        img_fpaths = batch['img_fpath']
        captions = batch['captions']

        images = [load_image(img) for img in image_fpaths]

        img_embeddings = img_model.encode(images)
        text_embeddings = text_model.encode(captions)

        #change the embeddings to tensor
        img_embeddings = torch.from_numpy(img_embeddings)   
        text_embeddings = torch.from_numpy(text_embeddings)

        #concat the two embeddings for the joint representation

        joint_embeddings = torch.cat((img_embeddings, text_embeddings), dim=1)

        #things to do:
            #1. prepare a small test file to test these embeddings on all the models
            #2. since we are not training the clip model we don't need much data - test on sample data
            #3. stack all the embeddings together and save it as .npy file for further analysis!!


        break
    exit()









    image = Image.open('/scratch365/apoudel/indosimcse/original_images/2021/09/13/874155d6bcc2a004c6f640952ed602eb.jpg')

    image_features = img_model.encode(image)

    #read the data


    print(image_features)
    exit()

# # Load the pre-trained CLIP model and processor
# # model = 'openai/clip-vit-base-patch32'
# model, preprocess = clip.load('ViT-B/32', device='cpu')
# clip_processor = CLIPProcessor.from_pretrained(model)

# # Load an example image
# image = Image.open('/scratch365/apoudel/indosimcse/original_images/2021/09/13/874155d6bcc2a004c6f640952ed602eb.jpg')

# # Preprocess the image
# inputs = clip_processor(images=image, return_tensors='pt')

# image_features = model.encode_image(inputs)

# print(input_features.shape)

# # print(inputs)

# uid,text,img_dir,img_fname
# f4a022dc-19a4-46e4-87d2-20ea4dda27f1,"rumus preset aplikasi lightroom[ bundle now preset ]  📍 jangan lupa klik like 💓 sayfollow kami ya untuk medapakan banyak rumus preset kece dan keren"" lainnya photo by @amirathallyapreset by @lightroom.contentintip paket preset terkece yuk di @juraganpreset.packmau preset yg praktis / mudah tinggal pakek ? tanpa ribet atur atur lagi ? cus langsung order saja ⬇⬇⬇ 📍price list preset📍( preset khusus android 📱 )15k, 47 preset 25k, 100 preset35k, 160 preset40k, 200 preset 50k, 300 preset 60k, 360 preset 75k, 460 preset100k, 600 preset135k, 860 preset( preset khusus ios, dan pc📱,💻 )20k, 40+ preset 35k, 87+ preset 55k, 147+ preset100k, 393 preset + gratis paket astetic, wedd, selfie, ootd, 150k, 519 preset, full pack, and feed, lengkap dengn paket preset, outdoor,indoor, beach,astetic,wedding,selfiewa : 081337639317ig : @lightroom.content ( klik link in bio )• yuk follow @lightroom.content supaya tidak  ketinggalan preset terbaru dan ter hits• save 📥 semoga membantu dilain waktu 😊• jangan lupa like❤ , supaya saya lebih   bersemangat membuat rumus preset untuk   kalian • request preset yang kamu suka lewat   comment dan dm • jangan lupa tag temen kamu yang suka   ngedit ✨pembayaran • dana & gopay• transfer ( semua jenis bank )• pulsa telkomsel • alfamartorder (wa) 081337639317( ig ) @lightroom.content ( klik link in bio )~~~~~~~~semoga membantu~~~~~~~#lightroom #presets #lr #presetslightroom #lightroompresets #lighroomindonesia #lightroomtutorials #lightroomcc #lighroommobile #lightroomedia#pemandangan #langit #indonesia #sawah #gunung #clouds #lightroom #lrpresets #presetbylightroomcontent",original_images/2021/09/19/,79bc6c04ecec2847674de279469c582d

# if __name__=="__main__":
#     main()