from torch.utils.data.dataset import Dataset
import torch

import numpy as np
from collections import Counter
import glob
import os
import unicodedata
import string

import pandas as pd

import pdb


class NameDataset(Dataset):
    def __init__(self):
        self.all_letters = string.ascii_letters + " .,;-'"
        self.n_letters = len(self.all_letters)
        self.all_categories = []
        self.category_lines = {}
        self.category_count = {}
        self.name_count = 0

        # Build the category_lines dictionary, a list of names per language
        for filename in self.find_files('names/*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.read_lines(filename)
            self.category_lines[category] = lines
            self.category_count[category] = len(lines)
            self.name_count += len(lines)

        self.n_categories = len(self.all_categories)

        # Dataframe with name and category column
        name_list = [(v, k) for k in [*self.category_lines] for v in self.category_lines[k]]
        self.name_df = pd.DataFrame(name_list, columns=['name', 'category'])
        self.name_df['name_len'] = self.name_df.name.str.len()
        # Create a column with equal length by adding spaces. For batch training.
        self.name_df['name_eq_len'] = self.name_df.name.apply(self.add_spaces, args=(self.name_df.name_len.max(),))

    def __len__(self):
        return self.name_count

    def __getitem__(self, idx):
        # name, category = self.name_df.iloc[idx]
        name, category = self.name_df.iloc[idx][['name_eq_len', 'category']]
        name_int = self.line_to_one_hot_vector(name[:10])
        category_int = self.all_categories.index(category)
        return name_int, category_int

    def add_spaces(self, input_str, target_n_spaces):
        n_spaces = target_n_spaces - len(input_str)
        # n_spaces = target_n_spaces - input_str.len()
        for i in range(n_spaces):
            input_str += ' '
        return input_str

    def find_files(self, path): return glob.glob(path)

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    # Read a file and split into lines
    def read_lines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicode_to_ascii(line) for line in lines]

    # Find letter index from all_letters, e.g. "a" = 0
    def letter_to_index(self, letter):
        return self.all_letters.find(letter)

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def line_to_one_hot_vector(self, line):
        tensor = np.zeros((len(line), self.n_letters))
        for li, letter in enumerate(line):
            tensor[li][self.letter_to_index(letter)] = 1
        return tensor