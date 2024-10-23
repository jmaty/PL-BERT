import csv

class TextCleaner:
    def __init__(self, symbols, pad='_'):
        self._symbols = symbols
        self._pad = pad
        assert len(self) == 81, f'Number of symbols must be 81 but it is {len(self)}'
        assert pad in symbols, f'Pad symbol ({pad}) is not included in symbols!'

    def __call__(self, text):
        indexes = []
        for c in text:
            try:
                indexes.append(self._symbols[c])
            except KeyError:
                # JMa:
                print(f'[!] Character  {c} not defined!\n    Utterance: {text}')
        return indexes

    def declean(self, indexes):
        return ''.join([self._symbols[i] for i in indexes])

    def __len__(self):
        return len(self._symbols)

    @property
    def symbols(self):
        return self._symbols

    @property
    def pad(self):
        return self._pad, self._symbols[self._pad]


def load_symbol_dict(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        symbol_dict = {row[0]: int(row[1]) for row in reader}
    return symbol_dict
