from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import random
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from NameDataset import NameDataset
from model import RNN

import pdb


learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
criterion = nn.NLLLoss()


# Find letter index from all_letters, e.g. "a" = 0
def tensorToLetter(tensor, all_letters):
    arr = tensor.detach().cpu().numpy()
    name = ''
    for i in range(arr.shape[0]):
        name += all_letters[arr[i].argmax()]
    return name


def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def train(rnn, optimizer, category_tensor, line_tensor, device):
    hidden = rnn.init_hidden().to(device)

    rnn.zero_grad()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor.type(torch.cuda.LongTensor))
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    optimizer.step()

    return output, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main():
    batch_size = 32
    epochs = 50
    current_loss = 0
    all_losses = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    name_dataset = NameDataset()
    dataloader = DataLoader(name_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    n_hidden = 128
    rnn = RNN(name_dataset.n_letters, n_hidden, name_dataset.n_categories, batch_size).to(device)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print('Start Training')

    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            # output, _ = rnn(Variable(data[0][0, 0:1].type(Tensor)), Variable(torch.zeros(1, n_hidden).type(Tensor)))
            name_tensor = Variable(data[0].transpose(0, 1).type(Tensor))
            category_tensor = Variable(data[1].type(Tensor))
            output, loss = train(rnn, optimizer, category_tensor, name_tensor, device)
            current_loss += loss
            # category = name_dataset.all_categories[int(category_tensor.detach().cpu().numpy()[0])]

        # Print epoch number, loss, name and prediction
        avg_loss = current_loss / (len(name_dataset) / batch_size)
        category = name_dataset.all_categories[int(category_tensor.detach().cpu().numpy()[0])]
        guess, guess_i = categoryFromOutput(output[0], name_dataset.all_categories)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('Epoch: %d (%s) %.4f %s / %s %s' % (
            epoch, timeSince(start), avg_loss, tensorToLetter(data[0][0], name_dataset.all_letters), guess, correct))

        # Add current loss avg to list of losses
        all_losses.append(avg_loss)
        current_loss = 0

        torch.save(rnn.state_dict(), "epoch_%d.pth" % epoch)


if __name__ == '__main__':
    main()
