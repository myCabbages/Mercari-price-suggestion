import pandas as pd
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

print('Reading the data......\n')
train = pd.read_csv('../data/train.tsv', sep='\t')
test = pd.read_csv('../data/test.tsv', sep='\t')

x_description = train['item_description']
total_vocab = ''
for i in range(len(x_description)):
    s = x_description[i]
    if s == None:
        total_vocab += ''
    else:
        total_vocab += s

vocab = set(total_vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}
embeds = nn.Embbedding(len(vocab),10)
lookup_tensor = torch.LongTensor([word_to_ix])

y_data = train['price']

def handle_missing(data):
    data.category_name.fillna(value='missing', inplace=True)
    data.brand_name.fillna(value='missing', inplace=True)
    data.item_description.fillna(value='missing', inplace=True)
    return data
