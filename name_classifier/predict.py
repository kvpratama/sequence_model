import numpy as np
import torch
from NameDataset import NameDataset
from model import RNN

import pdb


def letter_to_index(letter, all_letters):
    return all_letters.find(letter)


def line_to_one_hot_vector(line, n_letters, all_letters):
    tensor = np.zeros((len(line), n_letters))
    for li, letter in enumerate(line):
        tensor[li][letter_to_index(letter, all_letters)] = 1
    return tensor


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


if __name__ == '__main__':
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    name_dataset = NameDataset()
    # dataloader = DataLoader(name_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    n_hidden = 128
    rnn = RNN(name_dataset.n_letters, n_hidden, name_dataset.n_categories, batch_size).to(device)
    saved_state = torch.load('epoch_49.pth')
    rnn.load_state_dict(saved_state)

    predict('Jackson', rnn, name_dataset.n_letters, name_dataset.all_letters, name_dataset.all_categories, device)
    pdb.set_trace()
