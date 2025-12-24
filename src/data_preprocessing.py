import pandas as pd
import unicodedata
import re
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK", 3: "PAD"}
        self.n_words = 4  # Count SOS, EOS, UNK, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    # Separate punctuation
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()

def read_data(path, limit=None):
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    
    # Rename columns for easier access if needed, but iloc is safer
    pairs = []
    for i in range(len(df)):
        eng = normalizeString(str(df.iloc[i, 0]))
        fra = normalizeString(str(df.iloc[i, 1]))
        pairs.append([eng, fra])
    
    return pairs

def filterPair(p, max_length=15):
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length

def filterPairs(pairs, max_length=15):
    return [pair for pair in pairs if filterPair(pair, max_length)]

def prepareData(path, limit=None):
    pairs = read_data(path, limit)
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filterPairs(pairs)
    print(f"Trimmed to {len(pairs)} sentence pairs")
    
    input_lang = Lang("eng")
    output_lang = Lang("fra")
    
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index.get(word, UNK_token) for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = [SOS_token] + indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long)

class TranslationDataset(Dataset):
    def __init__(self, pairs, input_lang, output_lang):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        input_tensor = tensorFromSentence(self.input_lang, pair[0])
        target_tensor = tensorFromSentence(self.output_lang, pair[1])
        return input_tensor, target_tensor, pair[0], pair[1]

# Collate function to pad batches
def collate_fn(batch):
    input_tensors, target_tensors, _, _ = zip(*batch)

    input_lengths = [len(tensor) for tensor in input_tensors]
    target_lengths = [len(tensor) for tensor in target_tensors]

    input_tensors_padded = torch.nn.utils.rnn.pad_sequence(input_tensors, padding_value=PAD_token)
    target_tensors_padded = torch.nn.utils.rnn.pad_sequence(target_tensors, padding_value=PAD_token)

    return input_tensors_padded, target_tensors_padded, input_lengths, target_lengths

if __name__ == '__main__':
    # Test
    prepareData('data/eng_-french.csv', limit=100)
