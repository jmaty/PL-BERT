#!/usr/bin/python3

import pathlib
import json
import re
import os
import subprocess
import tempfile
import logging
from tpp import Tpp

class TppTtstool(Tpp):

    logger = logging.getLogger('TppTtstool')

    def __init__(self, lang,
                 sentence_pause=None, 
                 phrase_pause=None,
                 rm_primary_word_stress=True,
                 primary_word_stress="ˈ",
                 rm_secondary_word_stress=True,
                 secondary_word_stress="ˌ",
                 tts_tool_bin=None,
                 tts_tool_data=None,
                 punct=None):
        super().__init__(lang, sentence_pause,
                         phrase_pause,
                         rm_primary_word_stress,
                         primary_word_stress,
                         rm_secondary_word_stress,
                         secondary_word_stress)
        self._setup_bin(tts_tool_bin)
        self._setup_data(tts_tool_data)
        # Data
        self._data = None
        # Punctuation characters ([!] is reserved for glottal stop!)
        self._punct = punct if punct else '.,?:"!'
        # internal pause notation
        self._pause = '|'

    # Setup path to TTS tool binary
    def _setup_bin(self, tts_tool_bin_path):
        if tts_tool_bin_path is not None:
            # must exist and be executable
            tts_tool_bin_path = pathlib.Path(tts_tool_bin_path)
            assert tts_tool_bin_path.is_file(), f"tts_tool binary {tts_tool_bin_path} does not exist!"
            assert os.access(tts_tool_bin_path, os.X_OK), f"tts_tool binary {tts_tool_bin_path} is not executable!"
            # OK
            self._tts_tool_bin = pathlib.Path(tts_tool_bin_path)

    # Setup path to TTS tool data
    def _setup_data(self, tts_tool_data):
        self._tts_tool_data = tts_tool_data

    def _preprocess_phon(self, item):
        self.logger.debug('Item to preprocess: %s', item)
        if item.get('pause', False):
            item['text'] = self._pause
        self.logger.debug('Preprocessed item: %s', item)
        return item

    def _postprocess_phon(self, phntr):
        self.logger.debug('Text to postprocess: %s', phntr)
        # Normalize
        phntr = phntr.strip()
        # General phonetic postprocessing
        phntr = self.postprocess_phon(phntr)
        # - switch punctuation
        # phntr = re.sub('([^{0}])[ ]?([#$])([{0}])'.format(self._punct), '\\1\\3 \\2', phntr)
        phntr = re.sub('([^{0}])[ ]?([\|])([{0}])'.format(self._punct), '\\1\\3 \\2', phntr)
        # - remove space in front of punctuation
        phntr = re.sub('\s+([{0}])'.format(self._punct), '\\1', phntr)
        # Add final space, if there is such
        if not phntr.endswith(self._pause) :
            # phntr += ' #'
            phntr += f' {self._pause}'
        # Cross-sentence pauses
        if self._sentence_pause is not None:
            # Add cross-sentence pauses
            phntr = phntr.replace(self._pause, self._sentence_pause, 1)
            phntr = (phntr[::-1].replace(self._pause, self._sentence_pause, 1))[::-1]
        else:
            # Remove cross-sentence pauses
            phntr = phntr.replace(self._pause, '', 1)
            phntr = (phntr[::-1].replace(self._pause, '', 1))[::-1].strip()
        # Sentence-internal (phrase) pauses
        if self._phrase_pause is not None:
            # Add phrase pauses
            phntr = phntr.replace(self._pause, self._phrase_pause)
        else:
            # Remove phrase pauses
            phntr = phntr.replace(f'{self._pause} ', '')
        self.logger.debug('Postprocessed text:  %s', phntr)
        return phntr

    # ---------------------
    # SSML parsing
    #
    # - processes the input SSML text
    # - normalizes it
    #
    def ssml_parse(self, ssml_text):
        # -------
        # TPP SSML preskakuje <emphasis>...</emphasis>
        ssml_text = re.sub('<[/]?emphasis[^>]*>', '', ssml_text)
        # ------

        # Run the TPP
        with tempfile.NamedTemporaryFile(mode='wt', suffix='.ssml', encoding="utf8") as ifile:
            # Create the input file
            ifile.write(ssml_text)
            ifile.flush()
            # For orto parsing -marks is not important
            cmd = (self._tts_tool_bin, 'tpp', '-input', ifile.name, '-recipe', self._tts_tool_data, '-format', 'json', '-log-stderr')
            self.logger.debug(cmd)

            # Run the command
            err_string = ""
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
                try:
                    self._data = json.load(proc.stdout)
                except:
                    self.logger.warning("Reading empty JSON data!")
                    self._data = ()
                    err_string = '\n  '.join(l for l in proc.stderr.read().decode('utf8').split('\n'))
            # Fail
            if proc.returncode != 0:
                raise RuntimeError(f'tts_tool fail:\n  {err_string}')

    # ---------------------
    # Parsing to phonetic sentences 
    # - transcribes it into phonetic form
    # - splits it to sentences
    # - returns each sentence (as generator)
    def to_sentences_phon(self):
        # Process the parsed text
        phntr = ''
        for l in self._data:
            self.logger.debug('tts_tool: %s', l)
            l = self._preprocess_phon(l)
            if l['type'] == 'phone' and l['text']:
                if l.get('pause', False) and not phntr.endswith(self._pause) and phntr != '':
                    phntr += ' '
                phntr += l['text']
            elif l['type'] == 'whiteSpace':
                if not phntr:
                    phntr = self._pause
                phntr +=  ' '
            elif l['type'] == 'punctChar':
                phntr += l['text']
            elif l['type'] == 'word':
                if not phntr.endswith(' '):
                    phntr += ' '
            elif l['type'] in 'phraseBreak':
                if not phntr :
                    continue
                # Add punctuation, if there is any
                phntr += l.get('text', '')
            elif l['type'] in ('sentenceBreak', 'paragraphBreak'):
                if not phntr :
                    continue
                # Add punctuation, if there is any
                phntr += l.get('text', '')
                # Get the transcription
                self.logger.debug('%s', phntr)
                yield self._postprocess_phon(phntr)
                phntr = ''
            self.logger.debug('%s', phntr)
        # Get the remainder
        if phntr :
            yield self._postprocess_phon(phntr)

    # ---------------------
    # Parsing to ortographic sentences 
    # - splits it to sentences
    # - returns each sentence (as generator)
    def to_sentences_orto(self):
        # Process the parsed text
        text = ''
        for l in self._data:
            if l['type'] in ('sentenceBreak', 'paragraphBreak') and text:
                # Add punctuation, if there is any
                text += l.get('text', '')
                # Get the text and reset
                yield text.strip()
                text = ''
            elif l['type'] in ('word', 'whiteSpace', 'phraseBreak', 'punctChar'):
                text += l.get('text', '')
        # Get the remainder
        if text:
            yield text.strip()

    # ---------------------
    # Parsing to phonetic phrases 
    # - transcribes it into phonetic form
    # - splits it to phrases
    # - returns each phrase (as generator)
    def to_phrases_phon(self):
        # Process the parsed text
        phntr = ''
        for l in self._data:
            if   l['type'] == 'phone' and l['text']:
                if l.get('pause', False) and not phntr.endswith('#'):
                    phntr += ' '
                phntr += l['text']
            elif l['type'] == 'whiteSpace':
                if not phntr:
                    phntr = '$' if self._sentence_pause else '#'
                phntr +=  ' '
            elif l['type'] == 'punctChar':
                phntr += l['text']
            elif l['type'] == 'word':
                if not phntr.endswith(' '):
                    phntr += ' '
            elif l['type'] in ('phraseBreak', 'sentenceBreak', 'paragraphBreak'):
                if not phntr:
                    continue
                # Add punctuation, if there is any
                phntr += l.get('text', '')
                # Get the transcription
                yield self._postprocess_phon(phntr)
                phntr = ''
        # Get the remainder
        if phntr:
            yield self._postprocess_phon(phntr)

    # ---------------------
    # Parsing to ortographic phrases
    # - transcribes it into phonetic form
    # - splits it to phrases
    # - returns each phrase (as generator)
    def to_phrases_orto(self):
        # Process the parsed text
        text = ''
        for l in self._data:
            if l['type'] in ('phraseBreak', 'sentenceBreak', 'paragraphBreak') and text:
                # Add punctuation, if there is any
                text += l.get('text', '')
                # Get the text and reset
                yield text.strip()
                text = ''
            else:
                text += l.get('text', '')
        # Get the remainder
        if text:
            yield text.strip()

# # -------------
# # TEST:

# if __name__ == '__main__' :

#     infile   = 'text.ssml'
#     indata = open(infile, mode='rt').read()

#     for t in ssml_textnorm(indata, data() / 'frontend.json') :
#         print(t)
