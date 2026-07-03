import torch.nn as nn
from .embedding import EmbeddingLayer
from .encoder_block import EncoderBlock


class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, max_seq_length, num_heads, num_layers, ff_dim, dropout=0.1):
        super().__init__()

        # Combines raw token embeddings with positional encodings.
        # This gives the model the identity of the words AND their sequential order.
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            dropout=dropout
        )

        # stacked sequence of encoder blocks. 
        # nn.ModuleList is used, so pytorch for properly registers the parameters and updates them during backpropagation.
        self.layers = nn.ModuleList(
            [
                EncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
                
                for _ in range(num_layers)
            ]
        )

    def forward(self, input_ids):

        #token to embeddings
        x = self.embedding(input_ids) # (batch_size * num_choices, seq_len, embedding_dim)

        #sequentially pass the output of one encoder layer as the input to the next
        for layer in self.layers:
            x = layer(x)

        #shape of x: (batch_size * num_choices, seq_len, embedding_dim)
        return x