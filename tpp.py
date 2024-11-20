#!/usr/bin/python3

import pathlib
import json
import re
import os
import subprocess
import tempfile
import logging
from abc import ABC, abstractmethod

class Tpp(object):

    logger = logging.getLogger('Tpp')

    def __init__(self, lang, sentence_pause=None, phrase_pause=None,
                 rm_primary_word_stress=True, primary_word_stress="ˈ", rm_secondary_word_stress=False, secondary_word_stress="ˌ"):
        self._lang = lang
        # Long sentence leading/trailing pause ($) - typically in full phonetic alphabet
        self._sentence_pause = sentence_pause
        # Short sentence-internal pause
        self._phrase_pause = phrase_pause
        # Primary/secondary word stress
        self._rm_primary_word_stress = rm_primary_word_stress
        self._primary_word_stress = primary_word_stress
        self._rm_secondary_word_stress = rm_secondary_word_stress
        self._secondary_word_stress = secondary_word_stress

    # ---------------------
    # Parsing to phonetic sentences 
    # - transcribes it into phonetic form
    # - splits it to sentences
    # - returns each sentence (as generator)
    @abstractmethod
    def to_sentences_phon(self):
        raise NotImplementedError('Subclasses should implement this!')
    
    # ---------------------
    # Parsing to ortographic sentences 
    # - splits it to sentences
    # - returns each sentence (as generator)
    @abstractmethod
    def to_sentences_orto(self):
        raise NotImplementedError('Subclasses should implement this!')

    # ---------------------
    # Parsing to phonetic phrases 
    # - transcribes it into phonetic form
    # - splits it to phrases
    # - returns each phrase (as generator)
    @abstractmethod
    def to_phrases_phon(self):
        raise NotImplementedError('Subclasses should implement this!')
    
    # ---------------------
    # Parsing to ortographic phrases 
    # - transcribes it into phonetic form
    # - splits it to phrases
    # - returns each phrase (as generator)
    @abstractmethod
    def to_phrases_orto(self):
        raise NotImplementedError('Subclasses should implement this!')
    
    def postprocess_phon(self, ph_string):
        if self._rm_primary_word_stress:
            # It make no sense to use secondary word stress when primary is not used => both primary and secondary stresses are removed
            ph_string = ph_string.replace(self._primary_word_stress, "")
            ph_string = ph_string.replace(self._secondary_word_stress, "")
        elif self._rm_secondary_word_stress:
            ph_string = ph_string.replace(self._secondary_word_stress, "")
        return ph_string

# # -------------
# # TEST:

# if __name__ == '__main__' :

#     infile   = 'text.ssml'
#     indata = open(infile, mode='rt').read()

#     for t in ssml_textnorm(indata, data() / 'frontend.json') :
#         print(t)
