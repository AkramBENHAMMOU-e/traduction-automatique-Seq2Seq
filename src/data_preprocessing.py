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

    def trim(self, min_count=1, max_vocab_size=None):
        if min_count <= 1 and max_vocab_size is None:
            return

        kept = [(word, count) for word, count in self.word2count.items() if count >= min_count]
        kept.sort(key=lambda x: (-x[1], x[0]))
        if max_vocab_size is not None:
            kept = kept[: max(0, int(max_vocab_size))]

        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK", 3: "PAD"}
        self.n_words = 4

        for word, count in kept:
            self.word2index[word] = self.n_words
            self.word2count[word] = count
            self.index2word[self.n_words] = word
            self.n_words += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """
    Text normalization aligned with references/notebook (1).py (Tatoeba/ManyThings).
    - Lowercase + strip
    - Unicode normalize (NFKD)
    - Keep letters (incl. accented), digits, and basic punctuation
    - Collapse whitespace
    """
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^a-zA-Z0-9À-ÖØ-öø-ÿ'.,!?;:\-() ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalizeString_v2(s):
    """
    v2 normalization:
    - Lowercase + strip
    - Unicode normalize (NFKD)
    - Separate punctuation as tokens (e.g., "go." -> "go .")
    - Keep letters (incl. accented), digits, and basic punctuation
    - Collapse whitespace
    """
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"([.,!?;:()])", r" \1 ", s)
    s = re.sub(r"[-]", " - ", s)
    s = re.sub(r"[^a-zA-Z0-9À-ÖØ-öø-ÿ'.,!?;:\-() ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _get_normalizer(normalization):
    if callable(normalization):
        return normalization
    if str(normalization).lower() == "v2":
        return normalizeString_v2
    return normalizeString

def read_data(path, limit=None, normalizer=normalizeString):
    pairs = []

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        if limit is not None:
            df = df.head(limit)

        for i in range(len(df)):
            eng = normalizer(str(df.iloc[i, 0]))
            fra = normalizer(str(df.iloc[i, 1]))
            pairs.append([eng, fra])
        return pairs

    # Tatoeba / ManyThings format: "English<TAB>French<TAB>metadata"
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if limit is not None and line_idx >= limit:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            eng, fra = parts[0], parts[1]
            pairs.append([normalizer(eng), normalizer(fra)])

    return pairs


def read_parallel(src_path, tgt_path, limit=None, normalizer=normalizeString):
    pairs = []
    with open(src_path, "r", encoding="utf-8") as src_f, open(tgt_path, "r", encoding="utf-8") as tgt_f:
        for line_idx, (src_line, tgt_line) in enumerate(zip(src_f, tgt_f)):
            if limit is not None and line_idx >= limit:
                break
            src = normalizer(src_line.rstrip("\n"))
            tgt = normalizer(tgt_line.rstrip("\n"))
            if not src or not tgt:
                continue
            pairs.append([src, tgt])
    return pairs

def filterPair(p, max_length=15):
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length

def filterPairs(pairs, max_length=15):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def load_pairs(path, limit=None, max_length=15, normalization="v1"):
    normalizer = _get_normalizer(normalization)
    pairs = read_data(path, limit=limit, normalizer=normalizer)
    pairs = filterPairs(pairs, max_length=max_length)
    return pairs


def load_parallel_pairs(src_path, tgt_path, limit=None, max_length=15, normalization="v1"):
    normalizer = _get_normalizer(normalization)
    pairs = read_parallel(src_path, tgt_path, limit=limit, normalizer=normalizer)
    pairs = filterPairs(pairs, max_length=max_length)
    return pairs


def build_langs(
    train_pairs,
    input_vocab_size=None,
    output_vocab_size=None,
    min_word_freq=1,
    input_name="eng",
    output_name="fra",
):
    input_lang = Lang(input_name)
    output_lang = Lang(output_name)

    for src, tgt in train_pairs:
        input_lang.addSentence(src)
        output_lang.addSentence(tgt)

    input_lang.trim(min_count=min_word_freq, max_vocab_size=input_vocab_size)
    output_lang.trim(min_count=min_word_freq, max_vocab_size=output_vocab_size)
    return input_lang, output_lang


def prepareData(
    path,
    limit=None,
    max_length=15,
    input_vocab_size=None,
    output_vocab_size=None,
    min_word_freq=1,
    normalization="v1",
):
    pairs = load_pairs(path, limit=limit, max_length=max_length, normalization=normalization)
    print(f"Read {len(pairs)} sentence pairs")

    input_lang, output_lang = build_langs(
        pairs,
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        min_word_freq=min_word_freq,
    )
        
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
    prepareData('data/tatoeba/fra.txt', limit=100)
