import numpy as np


# Find letter index from all_letters, e.g. "a" = 0
def tensor_to_letter(tensor, all_letters):
    arr = tensor.detach().cpu().numpy()
    name = ''
    for i in range(arr.shape[0]):
        name += all_letters[arr[i].argmax()]
    return name


def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def letter_to_index(letter, all_letters):
    return all_letters.find(letter)


def line_to_one_hot_vector(line, n_letters, all_letters):
    tensor = np.zeros((len(line), n_letters))
    for li, letter in enumerate(line):
        tensor[li][letter_to_index(letter, all_letters)] = 1
    return tensor
