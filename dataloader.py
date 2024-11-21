#coding: utf-8

import logging
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from text_utils import TextCleaner, load_symbol_dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 tokenizer=None,
                 token_maps="token_maps.pkl",
                 word_separator=3039,
                 token_separator=" ",
                 token_mask="M",
                 symbol_dict_path="symbol_dict.txt",
                 max_mel_length=512,
                 word_mask_prob=0.15,
                 phoneme_mask_prob=0.1,
                 replace_prob=0.2):

        self.data = dataset
        self.max_mel_length = max_mel_length
        self.word_mask_prob = word_mask_prob
        self.phoneme_mask_prob = phoneme_mask_prob
        self.replace_prob = replace_prob
        self.text_cleaner = TextCleaner(load_symbol_dict(symbol_dict_path))

        self.word_separator = word_separator
        self.token_separator = token_separator
        self.token_mask = token_mask

        with open(token_maps, 'rb') as handle:
            self.token_maps = pickle.load(handle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        phonemes = self.data[idx]['phonemes']   # list of phonetic words
        input_ids = self.data[idx]['input_ids'] # list of word IDs

        # print(phonemes)
        # print(input_ids)

        words = []      # list of word IDs with each ID repeated word_length_in_phonemes times
        labels = ""     # phonetic sentence with ground-truth phonemes
        phoneme = ""    # (masked) phonetic sentence

        phoneme_list = ''.join(phonemes)    # phoneme string without word separators
        # print("phoneme_list:", phoneme_list)

        masked_index = []
        for phw, wid in zip(phonemes, input_ids):
            # Extend words with repeated word ID and a word separator
            # - word ID is repeated number of phonemes times
            # - each word ID is a list of word-piece IDs
            # - word separator is a appended as a 1-element list with its ID
            words.extend([wid]*len(phw) + [[self.word_separator]])
            # Add space between ground-truth phoneme words
            labels += phw + " "

            # Determine whether to mask the word
            if np.random.rand() < self.word_mask_prob:
                if np.random.rand() < self.replace_prob:
                    # Randomize or keep original phoneme based on probabilities
                    if np.random.rand() < (self.phoneme_mask_prob / self.replace_prob):
                        phoneme += ''.join(phoneme_list[np.random.randint(0, len(phoneme_list))] for _ in range(len(phw))) # randomized
                    else:
                        phoneme += phw
                else:
                    phoneme += self.token_mask * len(phw) # masked
                # Track masked indices
                masked_index.extend((np.arange(len(phoneme) - len(phw), len(phoneme))).tolist())
            else:
                phoneme += phw # ground-truth

            phoneme += self.token_separator # add space between phonetic words

        # phoneme: masked phoneme string with spaces between words
        # print('phoneme:', phoneme)
        mel_length = len(phoneme)
        masked_idx = np.array(masked_index)

        # if sentence is longer than max_mel_length, take a random slice of it
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            slice_end = random_start + self.max_mel_length
            # Slicing phoneme, words and labels
            phoneme = phoneme[random_start:slice_end]
            words = words[random_start:slice_end]
            labels = labels[random_start:slice_end]
            # adjust masked_index for the slice
            masked_index = [m - random_start for m in masked_idx if random_start <= m < slice_end]
        else:
            masked_index = masked_idx

        phoneme = self.text_cleaner(phoneme)
        # phoneme: (masked) phoneme IDs incl. punctuation and spaces between words
        # print('phoneme after cleaner:', phoneme)
        labels = self.text_cleaner(labels)
        # labels: ground-truth phoneme label IDs incl. punctuation and spaces between words

        # words: word IDs, each word = word ID repeated word_length_in_phonemes times
        # with word_separator ID in between words

        # print("words", words)
        # print("phoneme", phoneme)
        # print("labels", labels)

        words = [self.token_maps[tuple(w)]['token'] for w in words]

        assert len(phoneme) == len(words), f'phonemes: {len(phoneme)} vs words: {len(words)}'
        assert len(phoneme) == len(labels)

        phonemes = torch.LongTensor(phoneme)
        labels = torch.LongTensor(labels)
        words = torch.LongTensor(words)

        # Return:
        # - phonemes: list of (masked) phoneme IDs incl. punctuation and spaces between words
        # - words: list of word IDs;  each word = word ID repeated word_length_in_phonemes times
        # - labels: list of ground-truth phoneme label IDs incl. punctuation and spaces between words
        # - masked_index: list of masked phoneme indices
        return phonemes, words, labels, masked_index

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        

    def __call__(self, batch):
        batch_size = len(batch)

        # sort batch by length
        lengths = [b[1].shape[0] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        # get max length
        max_text_length = max(b[1].shape[0] for b in batch)

        words = torch.zeros((batch_size, max_text_length)).long()
        labels = torch.zeros((batch_size, max_text_length)).long()
        phonemes = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = []
        masked_indices = []

        for bid, (phoneme, word, label, masked_index) in enumerate(batch):
            text_size = phoneme.size(0)
            words[bid, :text_size] = word
            labels[bid, :text_size] = label
            phonemes[bid, :text_size] = phoneme
            input_lengths.append(text_size)
            masked_indices.append(masked_index)

        # Return:
        # - words: list of word IDs;  each word = word ID repeated word_length_in_phonemes times
        # - labels: list of ground-truth phoneme label IDs incl. punctuation and spaces between words
        # - phonemes: list of (masked) phoneme IDs incl. punctuation and spaces between words
        # - input_lengths: list of lengths of input phoneme strings
        # - masked_index: list of masked phoneme indices
        return words, labels, phonemes, input_lengths, masked_indices


def build_dataloader(df,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config=None,
                     dataset_config=None):

    if collate_config is None:
        collate_config = {}
    if dataset_config is None:
        dataset_config = {}
    dataset = FilePathDataset(df, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             # drop_last=(not validation),
                             drop_last=False,
                             collate_fn=collate_fn,
                             pin_memory=device != 'cpu')

    return data_loader
