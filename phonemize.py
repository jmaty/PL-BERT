import numpy as np
# import string
# from text_normalize import normalize_text, remove_accents

def phonemize(text, phonemizer, tokenizer, punctuation):
    # print('SENT:', text)
    text = text.replace('-', ' ')

    try:  # Phonify input text (sentence)
        phonemizer.ssml_parse(text.lower())
    except RuntimeError:
        print('[!] Phonemizing failed in tts_tool => skipping\n')
        print()
        raise

    ph_sentences = list(phonemizer.to_sentences_phon())
    if len(ph_sentences) > 1:
        print(f'[!] Phonemizing returned multiple sentences => using only the first one:\n{ph_sentences[0]}')
    # assert len(ph_sentences) == 1, f'Single sentence expected at the output of G2P:\n{ph_sentences}'
    # print('PHS: ', ph_sentences[0])

    words = split_punctuation(text.split(), punctuation)
    # print('WRD: ', words)
    ph_words = split_punctuation(ph_sentences[0].split(), punctuation)
    # print('PHW: ', ph_words)

    # Get word pieces and their IDs; append IDs to the sentence list
    # Go through all words in a sentence
    word_pieces_ids = [tokenizer.encode(word, add_special_tokens=False) for word in words]

    ## Get word piece IDs and phonetic pieces
    # inp_ids, inp_phwords = match_words_and_phwords(words, ph_words, tokenizer)

    # print("INP: ", word_pieces_ids)
    # print("OUT: ", ph_words)

    assert len(word_pieces_ids) == len(ph_words), f'Different lengths:\nwords={len(word_pieces_ids)}: {word_pieces_ids}\nphwords={len(ph_words)}: {ph_words}'
    return {'input_ids' : word_pieces_ids, 'phonemes': ph_words}


def split_punctuation(words, punctuations):
    result = []
    for word in words:
        # Odstraní všechny koncové interpunkční znaky
        stripped_word = word.rstrip(punctuations)
        # Zjistí, jaké interpunkční znaky byly odstraněny
        removed_punct_chars = word[len(stripped_word):]
        if stripped_word:
            result.append(stripped_word)
        # Přidá každý interpunkční znak jako samostatný prvek
        for char in removed_punct_chars:
            result.append(char)
    return result


def match_words_and_phwords(words, ph_words, tokenizer):
    # Init word piece IDs and phonetic pieces for the whole sentence
    all_w_piece_ids, all_ph_pieces= [], []

    # Go through all words in a sentence
    for word, ph_word in zip(words, ph_words):
        word_pieces_ids, ph_pieces = match_word_and_phonemes(word, ph_word, tokenizer)
        # all_w_piece_ids.extend(word_pieces_ids)
        all_w_piece_ids.append(word_pieces_ids)
        # all_ph_pieces.extend(ph_pieces)
        all_ph_pieces.append(ph_pieces)

    return all_w_piece_ids, all_ph_pieces


def match_word_and_phonemes(word, ph_word, tokenizer):
    # Get word pieces and their IDs; append IDs to the sentence list
    word_pieces_ids = tokenizer.encode(word, add_special_tokens=False)
    # Get phonetic pieces and append tehm to the sentence list
    word_pieces = tokenizer.tokenize(word)
    ph_pieces = split_ph_word(word_pieces, ph_word) if len(word_pieces_ids) > 1 else [ph_word]
    return word_pieces_ids, ph_pieces


def needleman_wunsch(s1, s2, match=1, mismatch=-1, gap=-1):
    """
    Needleman-Wunschův algoritmus pro zarovnání dvou řetězců s penalizací za mezery a neshody.

    Parametry:
    - s1, s2: Řetězce pro zarovnání
    - match: Skóre za shodu
    - mismatch: Penalizace za neshodu
    - gap: Penalizace za mezeru

    Návratová hodnota:
    - Zarovnané verze řetězců s mezerami, kde to je nutné
    """
    m, n = len(s1), len(s2)
    score = np.zeros((m+1, n+1), dtype=int)

    # Inicializace
    for i in range(m+1):
        score[i][0] = gap * i
    for j in range(n+1):
        score[0][j] = gap * j

    # Výpočet matice skóre
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                diag = score[i-1][j-1] + match
            else:
                diag = score[i-1][j-1] + mismatch
            up = score[i-1][j] + gap
            left = score[i][j-1] + gap
            score[i][j] = max(diag, up, left)

    # Traceback pro získání zarovnání
    align1, align2 = '', ''
    i, j = m, n
    while i > 0 or j > 0:
        current_score = score[i][j]
        if i > 0 and j > 0 and (s1[i-1] == s2[j-1] or current_score == score[i-1][j-1] + mismatch):
            align1 = s1[i-1] + align1
            align2 = s2[j-1] + align2
            i -= 1
            j -= 1
        elif i > 0 and (current_score == score[i-1][j] + gap):
            align1 = s1[i-1] + align1
            align2 = '-' + align2
            i -= 1
        else:
            align1 = '-' + align1
            align2 = s2[j-1] + align2
            j -= 1
    return align1, align2


def split_ph_word(word_pieces, ph_word):
    # Krok 1: Rekonstruujte původní slovo
    orig_word = ''
    for piece in word_pieces:
        if piece.startswith('##'):
            orig_word += piece[2:]
        else:
            orig_word += piece

    print("Original word:     ", orig_word)

    # Krok 2: Použití Needleman-Wunschova algoritmu
    aligned_word, aligned_phonetic = needleman_wunsch(orig_word, ph_word)
    print("Word pieces:       ", word_pieces)
    print("Aligned word:      ", aligned_word)
    print("Aligned phon. word:", aligned_phonetic)

    # Krok 3: Mapování word pieces na fonetické části
    ph_pieces = []
    idx = 0
    for piece in word_pieces:
        piece_clean = piece[2:] if piece.startswith('##') else piece
        piece_len = len(piece_clean)

        # Extrahujte odpovídající část z zarovnaného fonetického přepisu
        ph_piece = ''
        count = 0
        while count < piece_len and idx < len(aligned_word):
            if aligned_word[idx] != '-':
                count += 1
            if aligned_phonetic[idx] != '-':
                ph_piece += aligned_phonetic[idx]
            idx += 1
        ph_pieces.append(ph_piece)

    return ph_pieces

# # Test
# word_pieces = ["astro", "##nomie"]
# phonetic_transcription = "astronomije"
#
# phonetic_pieces = split_phonetic_transcription(word_pieces, phonetic_transcription)
# print("Fonetické pieces:", phonetic_pieces)
