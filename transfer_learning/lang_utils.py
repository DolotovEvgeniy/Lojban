import re
import unicodedata
from io import open

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)
MAX_LENGTH = 50
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        if isinstance(sentence, str):
            sentence = sentence.split(' ')
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_str(s):
    """Lowercase, trim, and remove non-letter characters"""
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_sentences(file):
    with open(file) as f:
        return [l.strip().lower().split() for l in f.read().lower().split('\n')]


def read_pairs(lang1, lang2, prefix, train=True):
    fname1 = "{}{}_{}.txt".format(prefix, "train" if train else "val", lang1)
    fname2 = "{}{}_{}.txt".format(prefix, "train" if train else "val", lang2)
    lang1_lines = read_sentences(fname1)
    lang2_lines = read_sentences(fname2)

    pairs = [(' '.join(s1), ' '.join(s2)) for s1, s2 in zip(lang1_lines, lang2_lines)]
    pairs = filter_pairs(pairs)
    return pairs


def read_langs(lang1, lang2, prefix, reverse=False):
    pairs_train = read_pairs(lang1, lang2, prefix, train=True)
    pairs_val = read_pairs(lang1, lang2, prefix, train=False)

    if reverse:
        pairs_train = [list(reversed(p)) for p in pairs_train]
        pairs_val = [list(reversed(p)) for p in pairs_val]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs_train, pairs_val


def filter_pair(p):
    return (len(p[0].split(' ')) < MAX_LENGTH and
            len(p[1].split(' ')) < MAX_LENGTH)
            # p[1].startswith(eng_prefixes))


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, prefix='', reverse=False):
    """
    -  Read text file and split into lines, split lines into pairs_train
    -  Normalize text, filter by length and content
    -  Make word lists from sentences in pairs_train"""

    input_lang, output_lang, pairs_train, pairs_val = read_langs(lang1, lang2, prefix, reverse)
    print("Read %s sentence pairs_train" % len(pairs_train))
    print("Trimmed to %s sentence pairs_train" % len(pairs_train))
    print("Counting words...")
    for pair in pairs_train:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    for pair in pairs_val:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs_train, pairs_val
