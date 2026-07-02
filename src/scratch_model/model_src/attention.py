import math
import torch
import torch.nn as nn

    
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim//num_heads

        self.Wq = nn.Linear(embedding_dim, embedding_dim)
        self.Wk = nn.Linear(embedding_dim, embedding_dim)
        self.Wv = nn.Linear(embedding_dim, embedding_dim)

        self.out_projection = nn.Linear(embedding_dim, embedding_dim)

    
    def forward(self, x, return_attention=False):

        # x shape: (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, _ = x.shape

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        
        # Split into multiple heads and transpose to: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale scores to prevent vanishing gradients in softmax (Dot-Product Scaling Factor)
        scores_scaled = scores / math.sqrt(self.head_dim)

        weights = torch.softmax(scores_scaled, dim=-1)
        
        # output shape after matmul: (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(weights, V)

        # Permute and flatten back to original embedding space: (batch_size, seq_len, embedding_dim)
        output = output.transpose(1, 2).contiguous()

        output = output.view(batch_size, seq_len, self.embedding_dim)
        output = self.out_projection(output)

        if return_attention:
            return output, weights
        else:
            return output 