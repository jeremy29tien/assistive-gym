# pytorch mlp for regression
import numpy as np
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import ReLU
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
import torch
import argparse


input_dim = 25  # 25 for raw features, 3 for linear features
output_dim = 7

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-output_dim].astype('float32')
        print(self.X.shape)
        self.y = df.values[:, -output_dim:].astype('float32')
        print(self.y.shape)
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), output_dim))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_val=0.1, n_test=0.2):
        # determine sizes
        test_size = round(n_test * len(self.X))
        val_size = round(n_val * len(self.X))
        train_size = len(self.X) - (val_size + test_size)
        # calculate the split
        return random_split(self, [train_size, val_size, test_size])

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(8, output_dim)
        xavier_uniform_(self.hidden3.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        return X

# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, val, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    val_dl = DataLoader(val, batch_size=1024, shuffle=False)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, val_dl, test_dl


# train the model
def train_model(train_dl, val_dl, model, num_epochs, learning_rate, patience, checkpoint_dir):
    # define the optimization
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    cum_loss = 0.0
    trigger_times = 0
    prev_val_loss = 100
    # enumerate epochs
    for epoch in range(num_epochs):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

            # print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            val_loss = evaluate_model(val_dl, model)
            if i % 100 == 99:
                print("epoch {}:{} loss {}, val_loss {}".format(epoch, i, cum_loss, val_loss))
                cum_loss = 0.0
                print("check pointing")
                torch.save(model.state_dict(), checkpoint_dir)

        # Early Stopping
        if val_loss > prev_val_loss:
            trigger_times += 1
            # print('trigger times:', trigger_times)
            if trigger_times >= patience:
                print("Early stopping.")
                print("Trained Weights:", model.state_dict())
                return
        else:
            trigger_times = 0

        prev_val_loss = val_loss
    print("finished training")
    print("Trained Weights:", model.state_dict())


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), output_dim))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--data_path', default='',
                        help="name and location for training data")
    parser.add_argument('--model_path', default='',
                        help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of training epochs")
    parser.add_argument('--lr', default=0.01, type=float, help="learning rate")
    parser.add_argument('--weight_decay', default=0.0, type=float, help="weight decay")
    parser.add_argument('--patience', default=10, type=int, help="number of iterations we wait before early stopping")
    args = parser.parse_args()

    # Set the random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)

    ## HYPERPARAMS ##
    num_epochs = args.num_epochs  # num times through training data
    lr = args.lr
    weight_decay = args.weight_decay
    patience = args.patience
    #################

    # prepare the data
    path = args.data_path
    print('pulling data from', path)
    train_dl, val_dl, test_dl = prepare_data(path)
    print("(Train, Val, Test):", len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset))
    # define the network
    model = MLP(input_dim)
    # train the model
    train_model(train_dl, val_dl, model, num_epochs, lr, patience, args.model_path)
    # evaluate the model
    mse = evaluate_model(test_dl, model)
    print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
    torch.save(model.state_dict(), args.model_path)
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
