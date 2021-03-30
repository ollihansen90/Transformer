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

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

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
    def __init(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        n_heads,
        device,
        forward_expansion,
        dropout,
        max_len
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    n_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, n_heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, n_heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, n_heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        n_heads,
        forward_expansion,
        dropout,
        device,
        max_len
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, n_heads, forward_expansion, dropout, device) 
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        n_heads=8,
        dropout=0,
        device="cuda",
        max_len=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            n_heads,
            device,
            forward_expansion,
            dropout,
            max_len
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            n_heads,
            device,
            forward_expansion,
            dropout,
            max_len
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones([trg_len, trg_len])).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

        def forward(self, src, trg):
            src_mask = self.make_src_mask(src)
            trg_mask = self.make_trg_mask(trg)
            enc_src = self.encoder(src, src_mask)
            out = self.decoder(trg, trg_mask)
            return out








