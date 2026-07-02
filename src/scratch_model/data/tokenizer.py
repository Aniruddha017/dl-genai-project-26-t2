import string

class Tokenizer:

    def __init__(self, lowercase=False):
        self.lowercase = lowercase
        self.punctuation = list(string.punctuation)

    def tokenize(self, text):

        if text is None:
            return []

        text = str(text)

        if self.lowercase:
            text = text.lower()

        # Add spaces around punctuation
        for symbol in self.punctuation:
            text = text.replace(symbol, f" {symbol} ")

        # Convert to list of tokens
        tokens = text.split()

        return tokens

    def encode(self, text, vocab):

        tokens = self.tokenize(text)

        return vocab.encode(tokens)


    def decode(self, ids, vocab):   

        return vocab.decode(ids)

