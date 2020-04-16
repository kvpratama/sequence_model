from torch.utils.data.dataset import Dataset
import torch

import numpy as np
from collections import Counter
import glob
import os
import unicodedata
import string

import pandas as pd

from utils import letter_to_index, line_to_one_hot_vector

import pdb


class NameDataset(Dataset):
    def __init__(self, test=False):
        self.all_letters = string.ascii_letters + " .,;-'"
        self.n_letters = len(self.all_letters)
        self.all_categories = []
        self.category_lines = {}
        self.category_count = {}
        self.name_count = 0
        self.test = test

        # Build the category_lines dictionary, a list of names per language
        for filename in self.find_files('./names/*.txt'):
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
        if self.test:
            name, category = self.name_df.iloc[idx][['name', 'category']]
        else:
            category = self.all_categories[np.random.randint(len(self.all_categories))]
            # print(category)
            category_df = self.name_df['name'][self.name_df.category == category]
            rand_idx = np.random.randint(category_df.shape[0])
            name = category_df.iloc[rand_idx]

        name_int = line_to_one_hot_vector(name, self.n_letters, self.all_letters)
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
