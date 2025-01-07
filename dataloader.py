#coding: utf-8

import logging
import pickle
import random

import numpy as np
import torch
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
        phwords = self.data[idx]['phonemes'] # list of phonetic words
        input_ids = self.data[idx]['input_ids'] # list of word IDs

        logger.debug("phwords (%d): %s", len(phwords), phwords)
        logger.debug("word IDs (%d): %s", len(input_ids), input_ids)

        word_ids = []      # list of word IDs with each ID repeated word_length_in_phonemes times
        gt_phsent = ""     # phonetic sentence with ground-truth phonemes
        masked_phsent = ""    # (masked) phonetic sentence

        phstring = ''.join(phwords)    # phoneme string without word separators

        masked_index = []
        for phw, wid in zip(phwords, input_ids):

            # Extend words with repeated word ID and a word separator
            # - word ID is repeated number-of-phonemes times
            # - each word ID is a list of word-piece IDs
            # - word separator is appended as a 1-element list with its ID
            # words.extend([wid]*len(phw) + [self.word_separator])
            if phw.startswith('##'): # Processing subword
                phw = phw.removeprefix('##') # Remove word-piece prefix
                word_ids.extend([wid]*len(phw)) # Do not prepend space
                gt_phsent += phw
            else: # Processing whole word or a beginning of a word
                # Add space between words
                word_ids.extend(([self.word_separator] if word_ids else []) + [wid]*len(phw))
                # Add space between ground-truth phonetic words
                gt_phsent += (self.token_separator if gt_phsent else "") + phw
                # Add space between masked phoneme words
                masked_phsent += self.token_separator if masked_phsent else ""

            # Determine whether to mask the word
            if np.random.rand() < self.word_mask_prob:
                if np.random.rand() < self.replace_prob:
                    # Randomize or keep original phoneme based on probabilities
                    if np.random.rand() < (self.phoneme_mask_prob / self.replace_prob):
                        masked_phsent += ''.join(phstring[np.random.randint(0, len(phstring))] for _ in range(len(phw))) # randomized
                    else:
                        masked_phsent += phw
                else:
                    masked_phsent += self.token_mask * len(phw) # mask the whole phonet. word
                # Track masked indices
                masked_index.extend((np.arange(len(masked_phsent) - len(phw), len(masked_phsent))).tolist())
            else:
                masked_phsent += phw # do not mask: add ground-truth phonetic word

        # masked_phsent: masked phoneme string with spaces between words
        logger.debug("masked phsent (%d): %s", len(masked_phsent), masked_phsent)

        mel_length = len(masked_phsent)
        masked_idx = np.array(masked_index)

        # if sentence is longer than max_mel_length, take a random slice of it
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            slice_end = random_start + self.max_mel_length
            # Slicing phoneme, words and labels
            masked_phsent = masked_phsent[random_start:slice_end]
            word_ids = word_ids[random_start:slice_end]
            gt_phsent = gt_phsent[random_start:slice_end]
            # adjust masked_index for the slice
            masked_index = [m - random_start for m in masked_idx if random_start <= m < slice_end]
        else:
            masked_index = masked_idx

        # masked phoneme IDs incl. punctuation and spaces between words
        masked_ph_ids = self.text_cleaner(masked_phsent)
        logger.debug("masked ph IDs (%d): %s", len(masked_ph_ids), masked_ph_ids)

        # ground-truth phoneme label IDs incl. punctuation and spaces between words
        gt_ph_ids = self.text_cleaner(gt_phsent)
        logger.debug("GT ph IDs (%d): %s", len(gt_ph_ids), gt_ph_ids)

        # word_ids: word IDs, each word_id = word ID repeated word_length_in_phonemes times
        # with word_separator ID in between words
        logger.debug("word IDs (%d): %s", len(word_ids), word_ids)

        # Map word tokens
        mapped_word_ids = [self.token_maps[w]['token'] for w in word_ids]
        logger.debug("mapped word IDs (%d): %s", len(mapped_word_ids), mapped_word_ids)

        assert len(masked_ph_ids) == len(gt_ph_ids),\
            f'phonemes: {len(masked_ph_ids)} vs labels: {len(gt_ph_ids)}'
        assert len(gt_ph_ids) == len(mapped_word_ids),\
            f'labels: {len(gt_ph_ids)} vs words: {len(word_ids)}\n{gt_phsent}\n{word_ids}'

        return (
            torch.LongTensor(masked_ph_ids), # list of (masked) phoneme IDs incl. punctuation and spaces between words
            torch.LongTensor(mapped_word_ids), # list of word IDs;  each word = word ID repeated word_length_in_phonemes times
            torch.LongTensor(gt_ph_ids), # list of ground-truth phoneme label IDs incl. punctuation and spaces between words
            masked_index, # list of masked phoneme indices
        )

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

        word_ids = torch.zeros((batch_size, max_text_length)).long()
        gt_ph_ids = torch.zeros((batch_size, max_text_length)).long()
        masked_ph_ids = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = []
        masked_indices = []

        for bid, (masked_ph_id, word_id, gt_ph_id, masked_index) in enumerate(batch):
            text_size = masked_ph_id.size(0)
            word_ids[bid, :text_size] = word_id
            gt_ph_ids[bid, :text_size] = gt_ph_id
            masked_ph_ids[bid, :text_size] = masked_ph_id
            input_lengths.append(text_size)
            masked_indices.append(masked_index)

        return (
            word_ids, # list of word IDs;  each word = word ID repeated word_length_in_phonemes times
            gt_ph_ids, # list of ground-truth phoneme label IDs incl. punctuation and spaces between words
            masked_ph_ids, # list of (masked) phoneme IDs incl. punctuation and spaces between words
            input_lengths, # list of lengths of input phoneme strings
            masked_indices, # list of masked phoneme indices
        )


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
