import numpy as np
import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext import datasets
# from dataprocessing import collate_fn_CBOW
from functools import partial
from torchtext.data.utils import get_tokenizer


# basic_eng_tokeniser = get_tokenizer('basic_english')


class PennDataset(Dataset):
    def __init__(self,text_loc,tranform = None):
        self.text_loc = text_loc
        self.tranform = tranform
        with open(self.text_loc,"r") as f:
            self.lines = f.readlines()
   
    def __len__(self):
        return len(self.lines)
   
    def __getitem__(self,idx):
        line = self.lines[idx]
        input_token_ids, output_token_id = self.tranform([line])
        if input_token_ids.dim() != 2 or output_token_id.dim() != 1:
            input_token_ids,output_token_id = torch.tensor([[1,1,1,1,1,1]]), torch.tensor([1])
        return input_token_ids, output_token_id




class CustomDataset(Dataset):
    def __init__(self,text_loc,encoding = "utf8",transform = None):
        self.text_loc = text_loc
        self.transform = transform
        with open(self.text_loc,"r",encoding=encoding,) as f:
            text = ' '.join([line.strip() for line in f.readlines()])
            self.lines = text.split(".")
   
    def __len__(self):
        return len(self.lines)


    def __getitem__(self,idx):
        if not self.tranform:
            return self.lines[idx]
        else:
            return self.transform(self.lines[idx])


def collate_into_batches_CBOW(batch):
    return torch.concat([x[0] for x in batch],dim=0), torch.concat([x[1] for x in batch],dim=0)




# vocabulary = torch.load("word2vec/saved_vocab/vocabulary.pth")
# CBOW_window_size = 3
# tranform = partial(collate_fn_CBOW,vocab = vocabulary,CBOW_window_size = CBOW_window_size)


# penn_dataset = PennDataset("data/datasets/PennTreebank/ptb.train.txt",tranform = tranform)




penn_dataset = CustomDataset("data/datasets/american_psycho/AMERICAN_PSYCHO.txt")
   
# print(penn_dataset[78])
 




