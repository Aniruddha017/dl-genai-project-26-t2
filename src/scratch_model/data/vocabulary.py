from collections import Counter


class Vocabulary:

    def __init__(self, min_freq=1):

        # Only words occuring for min_freq time are added to the vocabulary
        self.min_freq = min_freq

        # PAD token to make all sequences of the same length
        # UNK token for unknown words not found in the vocabulary
        # SEP seperates the question from the option
        self.word2idx = {"PAD": 0, "UNK": 1, "SEP": 2}

        self.idx2word = {0: "PAD", 1: "UNK", 2: "SEP"}


    def build(self, texts, tokenizer):

        counter = Counter()

        for text in texts:

            tokens = tokenizer.tokenize(text)

            counter.update(tokens)

        index = len(self.word2idx)

        for word, freq in counter.items():

            if word not in self.word2idx and freq >= self.min_freq:

                self.word2idx[word] = index
                self.idx2word[index] = word

                index += 1

    def encode(self, tokens):

        return [self.word2idx.get(token, 1) for token in tokens]


    def decode(self, ids):

        return [self.idx2word.get(idx, "UNK") for idx in ids]


    def __len__(self):

        return len(self.word2idx)
    