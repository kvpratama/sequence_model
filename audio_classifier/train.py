# Import comet_ml in the top of your file
from comet_ml import Experiment

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.utils.data import Dataset
from dataset import UrbanSoundDataset
from model import AudioLSTM

# import numpy as np
# import matplotlib.pyplot as plt

import pdb

def train(model, epoch):
    model.train()
    # hidden_state = model.init_hidden(hyperparameters["batch_size"])
    for batch_idx, (data, target) in enumerate(train_loader):
        # optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        # data = data.requires_grad_() #set requires_grad to True for training

        # pdb.set_trace()
        # torchaudio.save('savedata.wav', data[0].cpu().permute(1,0), 8000)
        # data = data.permute([0, 2, 1])

        # hidden_state = tuple([each.data for each in hidden_state])

        model.zero_grad()
        # pdb.set_trace()
        # output, hidden_state = model(data, hidden_state)
        # output, _ = model(data, hidden_state)
        output, hidden_state = model(data, model.init_hidden(hyperparameters["batch_size"]))
        # output = model(data)
        # output = output.permute(1, 0, 2) #original output dimensions are batchSizex1x10

        # loss = F.nll_loss(output[0], target) #the loss functions expects a batchSizex10 input
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if batch_idx % log_interval == 0: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))
            experiment.log_metric("train loss", loss, step=epoch*batch_idx * len(data))


def test(model, epoch):
    model.eval()
    # hidden_state = model.init_hidden(hyperparameters["batch_size"])
    correct = 0
    y_pred, y_target = [], []
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        # output = model(data)
        # data = data.permute([0, 2, 1])
        # hidden_state = tuple([each.data for each in hidden_state])
        # output, hidden_state = model(data, hidden_state)
        output, hidden_state = model(data, model.init_hidden(hyperparameters["batch_size"]))
        # output = output.permute(1, 0, 2)

        # pred = output.max(2)[1]  # get the index of the max log-probability
        pred = torch.max(output, dim=1).indices
        correct += pred.eq(target).cpu().sum().item()
        y_pred = y_pred + pred.tolist()
        y_target = y_target + target.tolist()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    experiment.log_metric("test accuracy", 100. * correct / len(test_loader.dataset), step=epoch)
    experiment.log_confusion_matrix(y_true=y_target, y_predicted=y_pred)


if __name__ == '__main__':

    # Create an experiment
    experiment = Experiment(project_name="UrbanSound8K", workspace="kdebugging")
    experiment.set_name("test .comet.config")
    hyperparameters = {"lr": 0.01, "weight_decay": 0.0001, "batch_size": 128, "in_feature": 168, "out_feature": 10}
    experiment.log_parameters(hyperparameters)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    csv_path = './data/UrbanSound8K/UrbanSound8K.csv'
    file_path = './data/UrbanSound8K/'

    train_set = UrbanSoundDataset(csv_path, file_path, range(1, 10))
    test_set = UrbanSoundDataset(csv_path, file_path, [10])
    print("Train set size: " + str(len(train_set)))
    print("Test set size: " + str(len(test_set)))

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)

    # model = Net()
    model = AudioLSTM(n_feature=hyperparameters["in_feature"], out_feature=hyperparameters["out_feature"])
    model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    clip = 5  # gradient clipping

    log_interval = 10
    for epoch in range(1, 41):
        # if epoch == 31:
        #     print("First round of training complete. Setting learn rate to 0.001.")
        # scheduler.step()
        # with experiment.train():
        train(model, epoch)
        # with experiment.test():
        test(model, epoch)
