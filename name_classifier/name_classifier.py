from __future__ import unicode_literals, print_function, division
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from NameDataset import NameDataset
from model import RNN

from utils import category_from_output, tensor_to_letter

import pdb


def train(rnn, optimizer, category_tensor, line_tensor, device, learning_rate, criterion):
    hidden = rnn.init_hidden().to(device)

    rnn.zero_grad()
    # optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor.type(torch.cuda.LongTensor))
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    # optimizer.step()

    return output, loss.item()


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main():
    batch_size = 1
    epochs = 50
    current_loss = 0
    all_losses = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
    criterion = nn.NLLLoss()

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
            output, loss = train(rnn, optimizer, category_tensor, name_tensor, device, learning_rate, criterion)
            current_loss += loss

        # Print epoch number, loss, name and prediction
        avg_loss = current_loss / (len(name_dataset) / batch_size)
        category = name_dataset.all_categories[int(category_tensor.detach().cpu().numpy()[0])]
        guess, guess_i = category_from_output(output[0], name_dataset.all_categories)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('Epoch: %d (%s) %.4f %s / %s %s' % (
            epoch, time_since(start), avg_loss, tensor_to_letter(data[0][0], name_dataset.all_letters), guess, correct))

        # Add current loss avg to list of losses
        all_losses.append(avg_loss)
        current_loss = 0

        torch.save(rnn.state_dict(), "epoch_%d.pth" % epoch)


if __name__ == '__main__':
    main()
