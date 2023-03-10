import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
data = pd.read_csv("data.csv",sep=';')
train,val = train_test_split(data,test_size=0.2)
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects

# TODO add batch size as flag and other flags
Train_data= ChallengeDataset(train,mode="train")
Val_data= ChallengeDataset(val,mode="val")

Train_loader = t.utils.data.DataLoader(Train_data,batch_size= 32)
Val_loader = t.utils.data.DataLoader(Val_data,batch_size= 32)
# create an instance of our ResNet model
model = model.ResNet()
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
loss = t.nn.BCELoss()
# set up the optimizer (see t.optim)
optimizer = t.optim.Adam(model.parameters(), lr=0.0005)
# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(model,loss,optimizer,Train_loader,Val_loader,3)
#
print(t.cuda.is_available())
# go, go, go... call fit on trainer
res = trainer.fit(16)
res=res.cpu()
# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')