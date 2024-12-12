# Download, preprocess and serve the enwik8 dataset as a DataLoader.
# Reference from original llama2.c and github.com/salesforce/awd-lstm-lm

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import zipfile
import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer_char import Dictionary

DATA_CACHE_DIR = "data"

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads the enwik8 dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the enwik8 dataset, unless it's already downloaded
    data_url = "http://mattmahoney.net/dc/enwik8.zip"
    data_filename = os.path.join(DATA_CACHE_DIR, "enwik8.zip")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")
    print("Download done.")

def pretokenize(args):
    enwik = os.path.join(DATA_CACHE_DIR, 'enwik8.zip')
    data = zipfile.ZipFile(enwik).read('enwik8')

    print('Length of enwik8: {}'.format(len(data)))
    # Reference from Salesforce Github
    # https://github.com/salesforce/awd-lstm-lm/blob/master/data/enwik8/prep_enwik8.py
    num_test_chars = 5000000

    train_data = data[: -2 * num_test_chars]
    valid_data = data[-2 * num_test_chars: -num_test_chars]
    test_data = data[-num_test_chars:]

    for fn, part in [('train.txt', train_data), ('valid.txt', valid_data), ('test.txt', test_data)]:
        print('{} will have {} bytes'.format(fn, len(part)))
        print('- Tokenizing...')
        part_str = ' '.join([str(c) if c != ord('\n') else '\n' for c in part])
        print('- Writing...')
        dest = os.path.join(DATA_CACHE_DIR, fn)
        f = open(dest, 'w').write(part_str)

class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source, corpus):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        # Main dictionary and corpus
        self.corpus = corpus
        if self.vocab_source == "enwik8":
            self.splitdict = {'train': self.corpus.train, 'val': self.corpus.valid, 'test': self.corpus.test}
        else:
            raise NotImplementedError()
  
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        # train/test split. let's use only shard 0 for test split, rest train
        shard = self.splitdict[self.split]
        while True:
            # open the dataset for reading but keep it on disk with memmap
            num_batches = len(shard) // self.max_seq_len
            num_batches -= 1  # drop the last partial batch
            assert num_batches > 0, "this shard is way too small? investigate."
            ixs = list(range(num_batches))
            rng.shuffle(ixs)
            for ix in ixs:
                start = ix * self.max_seq_len
                end = start + self.max_seq_len + 1
                # calling .astype will copy the data into a new numpy array, now in RAM
                chunk = shard[start:end]
                x = chunk[:-1]
                y = chunk[1:]
                yield x, y

# -----------------------------------------------------------------------------
# public interface functions

class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize"])
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "pretokenize":
        pretokenize(args)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
