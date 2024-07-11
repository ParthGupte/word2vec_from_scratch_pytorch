import numpy as np
import datetime
import torch
from torch import nn


class CBOW(nn.Module):
    def __init__(self,vocab_size: int,embed_size = 300):
        super(CBOW,self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_size
        )
        self.linear = nn.Linear(
            in_features = embed_size,
            out_features = vocab_size
        )
        self.softmax = nn.Softmax(dim=1)
        #just to save how the model was trained
        self.epochs_trained = 0
        self.window_size = None
        self.optimizer = None #torch.optim.SGD(self.parameters())
        self.loss_fn = None #nn.CrossEntropyLoss()
   
    def forward(self,input_token_ids:torch.Tensor): #input: (batch_size,seq_len)
        x = self.embeddings(input_token_ids) #x: (batch_size,seq_len,embedding_dim)
        x = x.mean(axis=1) #x: (batch_size,embedding_dim)
        x = self.linear(x) #x: (batch_size,vocab_size)
        return x #its in logits (batch_size,vocab_size)
   
    def predict_probs(self,input_token_ids:torch.Tensor):
        x = self(input_token_ids) #x: (batch_size,vocab_size)
        probs = self.softmax(x) #probs: (batch_size,vocab_size)
        return probs
   
    def predict(self,input_token_ids:torch.Tensor):
        probs = self.predict_probs(input_token_ids) #probs: (batch_size,vocab_size)
        output_token_ids = probs.argmax(dim=1) #output_token_ids: (batch_size,)
        return output_token_ids


