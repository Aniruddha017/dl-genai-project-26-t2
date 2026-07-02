import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length, dropout=0.1):
        super().__init__()
        
        #Word embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        #Positional embeddings
        self.positional_embedding = nn.Embedding(
            num_embeddings=max_seq_length,
            embedding_dim=embedding_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        
        batch_size, seq_len = input_ids.shape

        #Create positions
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        positions = positions.expand(batch_size, seq_len)

        token_embedding = self.token_embedding(input_ids)

        # Learned positional embeddings are directly added to the embeddings
        positional_embedding = self.positional_embedding(positions)

        embeddings = token_embedding + positional_embedding

        return self.dropout(embeddings)
    