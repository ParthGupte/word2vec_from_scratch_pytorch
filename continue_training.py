from training import *


model_num = 100
vocabulary = torch.load('word2vec/saved_vocab/vocabulary_psycho.pth')
cbow_loaded = torch.load("word2vec/saved_models/cbow_psycho_{}_epochs.pth".format(model_num))


epochs = 1
batch_size = 64
window_size = cbow_loaded.window_size #n on the left and n on the right of the centerword
loss_fn = cbow_loaded.loss_fn
optimizer = cbow_loaded.optimizer


penn_dataloader_CBOW = DataLoader(penn_dataset,batch_size,shuffle=True,collate_fn = partial(collate_fn_CBOW,vocab = vocabulary,CBOW_window_size = window_size))


t1 = time.time()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(penn_dataloader_CBOW,cbow_loaded,loss_fn,optimizer)
    print("Done!")
    # cbow_loaded.epochs_trained += 1


t2 = time.time()


print("Training Time: ",t2-t1,"secs")
torch.save(cbow_loaded,"word2vec/saved_models/cbow_psycho_{}_epochs.pth".format(cbow_loaded.epochs_trained))
