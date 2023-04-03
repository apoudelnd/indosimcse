import argparse
import os, sys
import random
from collections import defaultdict
import os
from itertools import islice
import regex
import numpy as np
from collections import OrderedDict
import json
from collections import Counter
import pandas as pd
import sys
import math
import re
# sys.path.append(".")
# sys.path.append("..")
# sys.path.append("../..")
import pandas as pd
import config
from langdetect import detect
import logging
from tqdm import tqdm

langs = list()
idtext = defaultdict(list)

UID = "uid"
TEXT = "text"
img_dir = "img_dir"
img_fname = "img_fname"

logger = logging.getLogger(__name__)

class DataReader:

    def __init__(self, dir_path):
        self.dir_path = dir_path
    
    def read_data(self):
        """
        returns the dictionary of ids and text of indonesian tweets
        {id: text}

        also writes the text into a .txt file -- needed to finetune simcse model

        """

        data = pd.read_csv(self.dir_path, lineterminator = '\n')
        countnan = 0
        with open(config.output_file, 'w') as fout:
            
            for i in tqdm(range(len(data['id']))):
            # for i in tqdm(range(5)):

                if isinstance(data['headline'][i], str) and isinstance(data['related_text'][i], str):
                    text = data['related_text'][i].lower()
                elif type(data['headline'][i]) == str and type(data['related_text'][i]) == float:
                    text = data['headline'][i].lower()
                elif type(data['headline'][i]) == float and type(data['related_text'][i]) == str:
                    text = data['related_text'][i].lower()
                else:
                    text = 'nan'

                if text != 'nan' and len(text.split(' ')) > 5:
                    try:
                        lang = detect(text)
                        langs.append(lang)
                        if lang == 'id':
                            text_out = re.sub(r"[\n\t]*", "", text)
                            text_out = re.sub('http[s]?://\S+', '', text_out)
                            fout.write(text_out.strip()+'\n')
                            #original_img_dir, original_img_filename
                            # idtext[data['id'][i]] = text_out.strip()
                            
                            #some filenames have .jpg, .png and some don't -- 
                            try:
                                data['original_img_filename'][i] = data['original_img_filename'][i].split('.')[0]
                            except:
                                pass

                            idtext[UID].append(data['id'][i])
                            idtext[TEXT].append(text_out.strip())
                            idtext[img_dir].append(data['original_img_dir'][i])
                            idtext[img_fname].append(data['original_img_filename'][i])

                    except:
                        language = "error"
                        # logger.warning(f"This row {i} throws and error:", text)

                else:
                    countnan+=1

        df = pd.DataFrame.from_dict(idtext)
        # df.reset_index(inplace=True)
        # df.columns =  [UID, TEXT, img_dir, img_fname]
        df.to_csv(config.pd_output_file, index = False)

  
        with open(config.out, 'w') as out_file:
            out_file.write(json.dumps(Counter(langs)))
        


        logger.info(Counter(langs))
        logger.info(countnan)

        # return idtext


def main():
        # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(config.log_level)

    DataReader(config.DATA_PATH).read_data()


if __name__=='__main__':
    main()


#remove http links and hashtags as well
