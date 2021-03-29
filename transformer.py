import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, n_heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.head_dim = embed_size//n_heads
        assert self.head_dim*self.n_heads==self.embed_size, "embed_size ist nicht durch n_heads teilbar"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(n_heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        n_values = values.shape[1]
        n_keys = keys.shape[1]
        n_query = query.shape[1]

        values = values.reshape(N, n_values, self.n_heads, self.head_dim)
        keys = keys.reshape(N, n_keys, self.n_heads, self.head_dim)
        query = query.reshape(N, n_query, self.n_heads, self.head_dim)

        energy = torch.einsum("nkhd,nqhd->nhqk", query, keys)

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e10"))

        attention = torch.softmax(energy/self.embed_size**(1/2), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", attention, values).reshape(
            N, n_query, self.n_heads*self.head_dim
        )

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(embed_size*forward_expansion, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask):
        attention = self.attention(values, keys, queries, mask)
        x = self.dropout(self.norm1(attention + queries))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init(self, )







