import pandas as pd
import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm
import torchmetrics
from torchmetrics import MeanAbsoluteError
from scipy import stats
import esm
from typing import Union

# Load ESM model and alphabet
esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()

# ESM alphabet
esm_alphabet = [
    '<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 
    'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 
    'Z', 'O', '.', '-', '<null_1>', '<mask>'
]

# Token to index dictionary
token2idx_dict = dict(zip(esm_alphabet, range(len(esm_alphabet))))

def pad_sequence(sequence: str, sequence_length: int, padding_token: str) -> str:
    padding_length = sequence_length - len(sequence)
    if padding_length > 0:
        return sequence + (padding_token * padding_length)
    else:
        return sequence[:sequence_length]

def get_fasta_dict(fasta_file: str, sequence_length: int, esm_alphabet: list) -> dict:
    padding_char = '<pad>'
    fasta_dict = {}
    head = None

    with open(fasta_file, 'r') as infile:
        for line in infile:
            if line.startswith(">"):
                if head is not None:
                    fasta_dict[head] = pad_sequence(fasta_dict[head], sequence_length, padding_char)
                head = line.strip().replace(">", "")
                fasta_dict[head] = ''
            elif head is not None:
                sequence = ''.join([char if char in esm_alphabet else '<unk>' for char in line.strip()])
                fasta_dict[head] += sequence
            else:
                raise ValueError("File format error: sequence data encountered before header")

        if head is not None:
            fasta_dict[head] = pad_sequence(fasta_dict[head], sequence_length, padding_char)

    return fasta_dict

def token2idx(token: str) -> int:
    return token2idx_dict.get(token, token2idx_dict['<unk>'])

def convert(seq: str, max_length: int = 200) -> np.ndarray:
    if not isinstance(seq, (str, list)) or not isinstance(max_length, int):
        raise TypeError(f"Expected seq to be a string or a list and max_length to be an int, got {type(seq)} and {type(max_length)}")

    if len(seq) > max_length:
        seq = seq[:max_length]
    
    tokens = [token2idx_dict['<cls>']]
    tokens += [token2idx_dict.get(tok, token2idx_dict['<unk>']) for tok in seq]
    tokens.append(token2idx_dict['<eos>'])
    
    padding_length = max_length + 2 - len(tokens)  # +2 for <cls> and <eos>
    tokens.extend([token2idx_dict['<pad>']] * padding_length)
    
    return np.array(tokens, dtype=int)

class SeqDataset(Dataset):
    def __init__(self, fasta_dict: dict, max_length: int = 200):
        super(SeqDataset, self).__init__()
        self.fasta_dict = fasta_dict
        self.keys = list(fasta_dict.keys())
        self.max_length = max_length

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.fasta_dict[self.keys[idx]]
        tokens = convert(seq, self.max_length)
        return torch.tensor(tokens, dtype=torch.long)

mask_tok_idx = token2idx_dict['<mask>']

def is_1hot_tensor(x: torch.Tensor) -> bool:
    return x.dim() >= 2 and (x.sum(dim=-1) == 1).all()

def apply_mask(x: torch.Tensor, mask: torch.Tensor, mask_tok_idx: int) -> torch.Tensor:
    assert is_1hot_tensor(x)
    K = x.size(-1)
    x_masked = x.detach().clone()
    x_masked[mask.squeeze(-1)] = F.one_hot(torch.tensor(mask_tok_idx), K).to(x)
    return x_masked
