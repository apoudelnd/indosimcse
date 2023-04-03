from torch.utils.data import Dataset, DataLoader
import pandas as pd


class CLIPDataset(Dataset):

    def __init__(self, uid, caption, img_dir, img_fname):
        self.uid = uid
        self.caption = caption
        self.img_dir = img_dir
        self.img_fname = img_fname
    
    def __len__(self):
        return len(self.caption)
    
    def __getitem__(self, item):
        
        uid = self.uid[item]
        img_dir = self.img_dir[item]
        img_fname = self.img_fname[item]
        captions = self.caption[item]
        img_fpath = img_dir + img_fname

        return {
            'uid': uid,
            'img_fpath': img_fpath,
            'captions': captions
        }

def create_data_loader(df, batch_size):
    ds = CLIPDataset(
        uid = df.uid.to_numpy(),
        caption = df.text.to_numpy(),
        img_dir = df.img_dir.to_numpy(),
        img_fname = df.img_fname.to_numpy(),
    )

    #fix me ! batch_size

    return DataLoader(
        ds, 
        batch_size = batch_size,
        num_workers = 2
    )
