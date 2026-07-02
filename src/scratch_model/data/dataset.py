import torch
from torch.utils.data import Dataset


class MCQDataset(Dataset):

    def __init__(self, dataframe, tokenizer, vocab, max_len=128):

        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len

        self.has_labels = "answer" in dataframe.columns

        self.answer_map = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4
        }

    def __len__(self):
        return len(self.df)

    def pad_sequence(self, ids):

        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids += [0] * (self.max_len - len(ids))

        return ids

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        question = row["prompt"]

        options = [
            row["A"],
            row["B"],
            row["C"],
            row["D"],
            row["E"]
        ]

        encoded_choices = []

        for option in options:
            # special SEP token to separate question from answer 
            text = question + " SEP " + option

            ids = self.tokenizer.encode(text, self.vocab)

            ids = self.pad_sequence(ids)

            encoded_choices.append(ids)
        
        # input_ids shape: (5, max_len)
        input_ids = torch.tensor(encoded_choices, dtype=torch.long)

        sample = {"input_ids": input_ids}

        if self.has_labels:
            # Convert string target into numerical target ('C' converted to 2)
            sample["label"] = torch.tensor(self.answer_map[row["answer"]], dtype=torch.long)

        return sample