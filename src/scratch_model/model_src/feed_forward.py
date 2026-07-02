import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_dim, dropout=0.1):
        super().__init__()

        # Expands the embedding dimension to a higher dimensional space (ff_dim)
        # This allows the model to process the attention outputs and learn more complex features
        self.fc1 = nn.Linear(embedding_dim, ff_dim)

        # GELU (Gaussian Error Linear Unit)
        #performs better than ReLU by providing smoother gradients.
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Projects the expanded dimensional space back down to the original embedding_dim
        self.fc2 = nn.Linear(ff_dim, embedding_dim)

    def forward(self, x):

        # Initial x shape: (batch_size, seq_len, embedding_dim)
        x = self.fc1(x) # Shape transitions to: (batch_size, seq_len, ff_dim)

        #Applies GELU activation
        x = self.gelu(x)
        
        #dropout to reduce overfitting
        x = self.dropout(x)
        
        x = self.fc2(x) # Shape returns to: (batch_size, seq_len, embedding_dim)
        
        return x
