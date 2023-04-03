import os
import numpy as np
import pandas as pd
import random
import math
from collections import defaultdict
import json

RANDOM_SEED = 123
random.seed(RANDOM_SEED)
n_choices = 30


list_of_tuples = list()
dict_of_all = defaultdict(list)

class DataReader():

    def __init__(self, data_path):
        self.data_path = data_path

    def read_data(self):
        self.df = pd.read_csv(self.data_path)
        for i, x in enumerate(list(self.df['vec'])):
            
            #some pre-processing -- should have done this earlier while saving the file itself
            x = x.replace('[','').replace(']','')
            ind_list = list()
            for splits in x.split(' '):
                if len(splits)>2:
                    ind_list.append(float(splits))
            
            #making it as a list of tuples -- compatible with the np.linalg.norm
            if np.array(ind_list).shape == (2,):
                list_of_tuples.append(tuple(ind_list))
            
            #we might not need this but it's okay for now!
            # dict_of_all[i]={
            #     'uid': self.df['uid'][i],
            #     'text': self.df['text'][i],
            #     'vecs': tuple(ind_list)
            # }
        
        # print(dict_of_all)
        return list_of_tuples


    def write_arts(self, jsonstring, file_path):
            jsonfile = open(file_path, "w")
            jsonfile.write(jsonstring)
            jsonfile.close

    def get_json_file(self, list_of_tuples, all_index_ls, all_labels):
        df_all = list()

        df_dict = dict()

        for i, index_ls in enumerate(all_index_ls):
            for ind, index in enumerate(index_ls):
                label = 1 if ind == len(index_ls)-1 else 0
                df_all.append({
                    'index': i,
                    'uid': self.df['uid'][index],
                    'text': self.df['text'][index],
                    'label': label
                    })
                
                # df_dict[i].append(

                # )
        
        if not os.path.exists(os.path.join(os.getcwd(), 'output')):
            new_dir = os.mkdir(os.path.join(os.getcwd(), 'output'))
        else:
            new_dir = os.path.join(os.getcwd(), 'output')

        df_json_c = json.dumps(df_all, indent = 4)
        self.write_arts(df_json_c, os.path.join(new_dir, 'out_text_fin.json'))


def get_random_points(list_of_tuples, n_choices):
    """
        choosing random centroids to start with -- at least get 10 different examples
    """
    return random.sample(range(len(list_of_tuples)), k=n_choices)




if __name__ == "__main__":
    data_path = '../model/simcse-embed/outputvec.csv'

    reader = DataReader(data_path)
    list_of_tuples = reader.read_data()

    print(len(list_of_tuples))
    orig_idx = set(range(len(list_of_tuples)))

    #get n random_indexes-- centroids
    random_indexes = get_random_points(list_of_tuples, n_choices)

    print(random_indexes)

    replace = orig_idx - set(random_indexes)

    random_points = list()

    for points in random_indexes:
        random_points.append(list_of_tuples[points])

    # list_with_no_randoms = [tmp for tmp in list_of_vecs if tmp not in random_points]
    # list_ = [tmp for tmp in list_of_vecs if tmp not in random_points]


    all_index_ls = list()
    all_labels = list()
    for point in random_indexes:

        index_ls = list()
        labels = list()

        #L2 distance
        dist = np.linalg.norm(np.array(list_of_tuples) - np.array(list_of_tuples[point]), axis=1)
        ind = np.argpartition(dist, 2)[1:3] #the first one is always 0

        index_ls.extend(list(ind)) #two closest points (indexes)
        index_ls.append(point) #one random centroid
        labels.extend(3*[0])

        all_idx_set = set(index_ls)

        replace_ind = orig_idx - all_idx_set

        x = random.choice(list(replace_ind))  #one random sentence -- most likely contrastive or the odd one

        index_ls.append(x)
        labels.extend([1])

        all_index_ls.append(index_ls)
        all_labels.append(labels)

        print("all_index", index_ls)
        print("labels", labels)
    reader.get_json_file(list_of_tuples, all_index_ls, all_labels)

