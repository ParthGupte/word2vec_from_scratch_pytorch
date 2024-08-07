from dataloading_custom import *
import torchtext
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from functools import partial
import sys


#vocab creation


basic_eng_tokeniser = get_tokenizer('basic_english')


def collate_fn_vocab(batch):
    lst_tokens = []
    for line in batch:
        lst_tokens.extend(basic_eng_tokeniser(line))
    return lst_tokens




# print(len(vocabulary))
# for item in penn_dataloader_vocab:
#     print(item)
#     break


#dataprocessing for training


def collate_fn_CBOW(batch,vocab:torchtext.vocab.Vocab,CBOW_window_size:int):
    batch_input, batch_output = [], []
    for line in batch:
        word_lst = basic_eng_tokeniser(line)
        token_id_seq = [vocab[word] for word in word_lst]
        if len(token_id_seq) < CBOW_window_size*2 +1:
            continue
        for idx in range(len(token_id_seq)-(CBOW_window_size*2+1)):
            window_token_ids = token_id_seq[idx:idx+CBOW_window_size*2+1]
            output_token_id = window_token_ids.pop(CBOW_window_size)
            input_token_ids = window_token_ids
            batch_input.append(input_token_ids)
            batch_output.append(output_token_id)
   
    batch_input = torch.tensor(batch_input,dtype=torch.long).to("cuda")
    batch_output = torch.tensor(batch_output,dtype=torch.long).to("cuda")
    return batch_input, batch_output








def main():
    penn_dataloader_vocab = DataLoader(penn_dataset,collate_fn = collate_fn_vocab)
    vocabulary = build_vocab_from_iterator(penn_dataloader_vocab)
    vocabulary.append_token("<unk>")
    vocabulary.set_default_index(vocabulary["<unk>"])
    torch.save(vocabulary,'saved_vocab/vocabulary_RIP.pth')


if __name__ == "__main__":
    main()
