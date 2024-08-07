from models import *
from dataprocessing import *
from dataloading_custom import *
import time


# cbow = CBOW(30,10)
# input_tensor = torch.tensor([[1,2,3],[0,9,8]],dtype=torch.long)
# out = cbow.predict(input_tensor)
# print(out)
# print(out.shape)




def train_loop(dataloader:DataLoader, model:nn.Module,loss_fn:nn.CrossEntropyLoss, optimizer):
    # size = len(dataloader)
    # print(size)


    model.train()
    for batch, (X,y) in enumerate(dataloader):
        # print(batch)
        pred = model(X)
        loss = loss_fn(pred,y)


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X) + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}]")
    model.epochs_trained += 1


def main():
    vocabulary = torch.load('saved_vocab/vocabulary_RIP.pth')
    cbow = CBOW(len(vocabulary))
    cbow.to("cuda")
    epochs = 100
    batch_size = 128
    window_size = 3 #n on the left and n on the right of the centerword
    loss_fn = nn.CrossEntropyLoss().to("cuda")
    optimizer = torch.optim.SGD(cbow.parameters())


    cbow.window_size = window_size
    cbow.loss_fn = loss_fn
    cbow.optimizer = optimizer


    penn_dataloader_CBOW = DataLoader(penn_dataset,batch_size,True,collate_fn=partial(collate_fn_CBOW,vocab=vocabulary,CBOW_window_size=window_size))
    # penn_dataloader_CBOW = DataLoader(penn_dataset,batch_size,True,collate_fn=collate_into_batches_CBOW,num_workers=2)
    t1 = time.time()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(penn_dataloader_CBOW,cbow,loss_fn,optimizer)
        if (t+1)%10 == 0:
            print("Saving model")
            torch.save(cbow,"saved_models/cbow_RIP_{}_epochs.pth".format(t+1))
        print("Done!")
    t2 = time.time()


    print("Training Time: ",t2-t1,"secs")
    torch.save(cbow,"saved_models/cbow_RIP_{}_epochs.pth".format(epochs))


if __name__ == '__main__':
    main()

