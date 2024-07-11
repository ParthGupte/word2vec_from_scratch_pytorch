from models import *
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torchtext; torchtext.disable_torchtext_deprecation_warning()


model_num = 100
cbow_loaded = torch.load("word2vec/saved_models/cbow_psycho_{}_epochs.pth".format(model_num))
vocabulary = torch.load('word2vec/saved_vocab/vocabulary_psycho.pth')
embedding_layer_wts = list(cbow_loaded.parameters())[0]
# print(embedding_layer_wts[0].shape)
word_lst = vocabulary.get_itos()


def plot_embeddings_2D(embedding_array,word_lst):
    pca = PCA(n_components = 2)
    embeddings_2D = pca.fit_transform(embedding_array)


    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    x = embeddings_2D[:,0]
    y = embeddings_2D[:,1]
    plt.scatter(x,y)
    for i, word in enumerate(word_lst):
        ax.text(x[i],y[i],word)
    plt.savefig("word2vec/word_embedding_plots/cbow_{}_epochs_embeddings_2D_plot".format(model_num))
    plt.show()
    plt.close()


def closest_vector(vector:torch.tensor,embedding_layer_wts:torch.tensor=embedding_layer_wts):
    cosine = nn.CosineSimilarity()
    cos = cosine(vector,embedding_layer_wts)
    idx = cos.argmax()
    return idx


def closest_n_vectors(vector:torch.tensor,n=3,embedding_layer_wts:torch.tensor=embedding_layer_wts):
    cosine = nn.CosineSimilarity()
    cos = cosine(vector,embedding_layer_wts)
    L = cos.argsort()
    return L[:n]


def similar_word(word:str,vocab = vocabulary,embedding_layer_wts=embedding_layer_wts):
    word_idx = vocab[word]
    if word_idx == vocab.get_default_index():
        return "<unk>"
    word_vec = embedding_layer_wts[word_idx]
    cosine = nn.CosineSimilarity()
    cos = cosine(word_vec,embedding_layer_wts)
    cos[word_idx] = -100
    idx = cos.argmax()
    sim_word = vocab.lookup_token(idx)
    return sim_word


class Token():
    def __init__(self,word:str,vocab=vocabulary,embeddings = embedding_layer_wts):
        self.vocab = vocabulary
        self.embeddings = embeddings
        self.word = word
        self.word_idx = self.vocab[word]
        if self.word_idx == vocab.get_default_index():
            raise Exception("Word is not part of Vocabulary")
        self.word_vec = self.embeddings[self.word_idx]
    def __add__(self,other):
        if isinstance(other,Token):
            sum_vec = self.word_vec + other.word_vec
            closest_3 = closest_n_vectors(sum_vec,3)
            for i in closest_3:
                if i != self.word_idx and i != other.word_idx:
                    sum_idx = i
            sum_word = self.vocab.lookup_token(sum_idx)
            sum_token = Token(sum_word,self.vocab,self.embeddings)
            return sum_token
        else:
            raise Exception("Cant add these")
   
    def __sub__(self,other):
        if isinstance(other,Token):
            sub_vec = self.word_vec - other.word_vec
            closest_3 = closest_n_vectors(sub_vec,3)
            for i in closest_3:
                if i != self.word_idx and i != other.word_idx:
                    sub_idx = i
            sub_word = self.vocab.lookup_token(sub_idx)
            sub_token = Token(sub_word,self.vocab,self.embeddings)
            return sub_token
        else:
            raise Exception("Cant subtract these")
       
    def __str__(self):
        return self.word
       
#visualisation
def main():
    embedding_array = embedding_layer_wts.detach().numpy().copy()
    embedding_array[0] = np.zeros(embedding_array[0].shape)
    patrick = Token("patrick")
    blonde = Token("blonde")
    sex = Token("sex")
    woman = Token("woman")
    money = Token("money")
    emotion = Token("emotion")
    blood = Token("blood")
    I = Token("i")
    greed = Token("greed")
    kill = Token("kill")
    dinner = Token("dinner")
    drink = Token("drink")
    price = Token("price")
    routine = Token("routine")
    homeless = Token("homeless")
    dorsia = Token("dorsia")
    print(blonde+sex)
    print(patrick+dorsia)
    # plot_embeddings_2D(embedding_array,word_lst)
    print(similar_word("dorsia"))
    # print(similar_word("woman"))
    # print(similar_word("man"))
    # print(similar_word("money"))
    # print(similar_word("emotion"))
    # print(similar_word("greed"))  
    # print(similar_word("patrick"))
    # print(similar_word("kill"))


     








if __name__ == '__main__':
    main()
