import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from NameDataset import NameDataset
from model import RNN
from utils import category_from_output, line_to_one_hot_vector

import pdb


# Just return an output given a line
def evaluate(rnn, line_tensor, device):
    hidden = rnn.init_hidden().to(device)
    line_tensor = torch.from_numpy(line_tensor).float().to(device)

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i:i+1], hidden)

    return output


def predict(input_line, rnn, n_letters, all_letters, all_categories, device, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(rnn, line_to_one_hot_vector(input_line, n_letters, all_letters), device)

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

def plot_confusion(rnn, dataset, device, Tensor):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)

    # Keep track of correct guesses in a confusion matrix
    confusion = np.zeros((len(dataset.all_categories), len(dataset.all_categories)))

    # Go through a bunch of examples and record which are correctly guessed
    for i, data in enumerate(dataloader):
        name_tensor = Variable(data[0].transpose(0, 1).type(Tensor))
        category_tensor = Variable(data[1].type(Tensor))

        # category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(rnn, data[0][0].numpy(), device)
        guess, guess_i = category_from_output(output, dataset.all_categories)
        category = name_dataset.all_categories[int(category_tensor.detach().cpu().numpy()[0])]
        category_i = dataset.all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(len(dataset.all_categories)):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + name_dataset.all_categories, rotation=90)
    ax.set_yticklabels([''] + name_dataset.all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


if __name__ == '__main__':
    batch_size = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    name_dataset = NameDataset()
    # dataloader = DataLoader(name_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    n_hidden = 128
    rnn = RNN(name_dataset.n_letters, n_hidden, name_dataset.n_categories, batch_size).to(device)
    saved_state = torch.load('epoch_0.pth')
    rnn.load_state_dict(saved_state)

    plot_confusion(rnn, name_dataset, device, Tensor)

    predict('Jackson', rnn, name_dataset.n_letters, name_dataset.all_letters, name_dataset.all_categories, device)
    pdb.set_trace()
