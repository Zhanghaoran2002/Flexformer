import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torchtext.data.utils import get_tokenizer

class ListOpsDataset(Dataset):
    def __init__(self, file_path, vocab=None, max_length=2000, build_vocab=False):
        self.samples = []
        self.max_length = max_length
        self.tokenizer = get_tokenizer("basic_english")
        with open(file_path, encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                source, target = line.strip().split('\t')
                source = source.replace(']', 'X').replace('(', '').replace(')', '')
                self.samples.append((source, int(target)))
        print(f"Loaded {len(self.samples)} samples from {file_path}")
        if build_vocab:
            counter = Counter()
            for source, _ in self.samples:
                counter.update(self.tokenizer(source))
            self.vocab = {token: idx for idx, token in enumerate(counter.keys())}
        else:
            self.vocab = vocab

    def encode(self, text):
        tokens = self.tokenizer(text)
        ids = [self.vocab.get(token, 0) for token in tokens][:self.max_length]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        source, target = self.samples[idx]
        input_ids = self.encode(source)
        return input_ids, torch.tensor(target, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True)
    targets = torch.stack(targets)
    return {'inputs': inputs_padded, 'targets': targets}

def get_datasets(n_devices, task_name, data_dir=None, batch_size=256, max_length=2000):
    if batch_size % n_devices:
        raise ValueError(f"Batch size {batch_size} isn't divided evenly by n_devices {n_devices}")

    train_path = os.path.join(data_dir, f"{task_name}_train.tsv")
    val_path = os.path.join(data_dir, f"{task_name}_val.tsv")
    test_path = os.path.join(data_dir, f"{task_name}_test.tsv")

    train_dataset = ListOpsDataset(train_path, max_length=max_length, build_vocab=True)
    vocab = train_dataset.vocab

    val_dataset = ListOpsDataset(val_path, vocab=vocab, max_length=max_length)
    test_dataset = ListOpsDataset(test_path, vocab=vocab, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, vocab

