from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from data_reader import DataReader, PAD_TOKEN, EOS_TOKEN, GO_TOKEN


class PTBDataReader(DataReader):
    """
    DataReader used to read in the Penn Treebank dataset.
    """

    UNKNOWN_TOKEN = "<unk>"  # already defined in the source data

    DROPOUT_WORDS = {"a", "an", "the"}
    DROPOUT_PROB = 0.0

    REPLACEMENTS = {"there": "their", "their": "there"}
    REPLACEMENT_PROB = 0.25

    def __init__(self, config, train_path):
        super(PTBDataReader, self).__init__(
            config, train_path, special_tokens=[PAD_TOKEN, GO_TOKEN, EOS_TOKEN])

        self.UNKNOWN_ID = self.token_to_id[PTBDataReader.UNKNOWN_TOKEN]

    def read_samples_by_string(self, path):

        for line in self.read_tokens(path):
            source = []
            target = []

            for token in line:
                target.append(token)

                # Randomly dropout some words from the input.
                dropout_word = (token in PTBDataReader.DROPOUT_WORDS and
                                random.random() < PTBDataReader.DROPOUT_PROB)
                replace_word = (token in PTBDataReader.REPLACEMENTS and
                                random.random() <
                                PTBDataReader.REPLACEMENT_PROB)

                if replace_word:
                    source.append(PTBDataReader.REPLACEMENTS[token])
                elif not dropout_word:
                    source.append(token)

            yield source, target

    def unknown_token(self):
        return PTBDataReader.UNKNOWN_TOKEN

    def read_tokens(self, path):
        with open(path, "r") as f:
            for line in f:
                yield line.rstrip().lstrip().split()


class MovieDialogReader(DataReader):
    """
    DataReader used to read and tokenize data from the Cornell open movie
    dialog dataset.
    """

    UNKNOWN_TOKEN = "UNK"

    DROPOUT_TOKENS = {"a", "an", "the", "'ll", "'s", "'m", "'ve", "'re"}

    REPLACEMENTS = {
        "there": "their",
        "their": "there",
        "there": "they 're",
        "they 're": "there",
        "their": "they 're",
        "they 're": "their",
        "then": "than",
        "than": "then",
        "your": "you 're",
        "you 're": "your",
        "ill": "i 'll",
        "i 'll": "ill",
        "its": "it' s",
        "it 's": "its"
    }

    def __init__(self, config, train_path=None, token_to_id=None,
                 dropout_prob=0.0, replacement_prob=0.25, dataset_copies=2):
        super(MovieDialogReader, self).__init__(
            config, train_path=train_path, token_to_id=token_to_id,
            special_tokens=[
                PAD_TOKEN, GO_TOKEN, EOS_TOKEN,
                MovieDialogReader.UNKNOWN_TOKEN],
            dataset_copies=dataset_copies)

        self.dropout_prob = dropout_prob
        self.replacement_prob = replacement_prob

        self.UNKNOWN_ID = self.token_to_id[MovieDialogReader.UNKNOWN_TOKEN]

    def read_samples_by_string(self, path):
        for tokens in self.read_tokens(path):
            source = []
            target = []

            i = 0
            while i < len(tokens):
                target.append(tokens[i])

                # Randomly dropout some words from the input.
                dropout_token = (tokens[i] in MovieDialogReader.DROPOUT_TOKENS and
                                random.random() < self.dropout_prob)
                replace_token = (tokens[i] in MovieDialogReader.REPLACEMENTS and
                                random.random() < self.replacement_prob)
                replace_two_tokens = (i + 1 < len(tokens) and tokens[i] + " " + tokens[i + 1] in MovieDialogReader.REPLACEMENTS and
                                random.random() < self.replacement_prob)

                if replace_token:
                    for token in MovieDialogReader.REPLACEMENTS[tokens[i]].split():
                        source.append(token)
                elif replace_two_tokens:                    
                    for token in MovieDialogReader.REPLACEMENTS[tokens[i] + " " + tokens[i + 1]].split():
                        source.append(token)
                    target.append(tokens[i + 1])
                    i = i + 1
                elif not dropout_token:
                    source.append(tokens[i])

                i = i + 1

            yield source, target

    def unknown_token(self):
        return MovieDialogReader.UNKNOWN_TOKEN

    def read_tokens(self, path):
        with open(path, "r") as f:
            for line in f:
                yield line.lower().strip().split()

