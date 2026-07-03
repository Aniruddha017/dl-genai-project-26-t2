import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoder


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length, num_heads, num_layers, ff_dim, dropout=0.1):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size=vocab_size, 
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(embedding_dim, 1)


    def forward(self, input_ids):

        # input_ids shape: (batch_size, num_choices=5, seq_len)
        batch_size, num_choices, seq_len = input_ids.shape

        # Flatten the batch and choice dimensions together so the encoder treats each multiple-choice option as an independent sequence.
        # New shape: (batch_size * 5, seq_len)
        input_ids = input_ids.reshape(batch_size * num_choices, seq_len)
        
        #encode
        x = self.encoder(input_ids) #(batch_size * 5, seq_len, embedding_dim)

        #mean pooling
        x = x.mean(dim=1) #(batch_size * 5, embedding_dim)

        x = self.dropout(x)

        # Project down to a single raw score (logit) per choice
        logits = self.classifier(x) #(batch_size * 5, 1)

        # back to shape (batch, 5)
        logits = logits.view(batch_size, num_choices)

        return logits

