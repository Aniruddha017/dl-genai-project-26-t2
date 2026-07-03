import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward

class EncoderBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(embedding_dim, num_heads)

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.ffn = FeedForward(embedding_dim, ff_dim, dropout)

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        
        attention_output = self.attention(x)
        
        # Residual connection 1 followed by Layer Normalization (Post-LN)
        x = self.norm1(x + self.dropout(attention_output))

        ffn_output = self.ffn(x)
        
        # Residual connection 2 followed by Layer Normalization
        x = self.norm2(x + self.dropout(ffn_output))

        return x 
    
    
