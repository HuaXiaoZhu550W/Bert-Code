import torch.nn as nn
from .AddNorm import AddNorm
from .PositionWiseFFN import PositionWiseFFN
from .MultiHeadAttention import MultiHeadAttention
from .BertEmbedding import BertEmbedding


class EncoderBlock(nn.Module):
    """bert编码器块"""
    def __init__(self, embed_dim, num_heads, ffn_hiddens, dropout, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.addnorm1 = AddNorm(normalized_shape=embed_dim, dropout=dropout)
        self.ffn = PositionWiseFFN(ffn_inputs=embed_dim, ffn_hiddens=ffn_hiddens, ffn_outputs=embed_dim)
        self.addnorm2 = AddNorm(normalized_shape=embed_dim, dropout=dropout)

    def forward(self, queries, keys, values, mask):
        attention = self.attention(queries, keys, values, mask)
        X = self.addnorm1(queries, attention)
        return self.addnorm2(X, self.ffn(X))


class BertEncoder(nn.Module):
    """ bert编码器"""
    def __init__(self, vocab_size, embed_dim, num_heads, ffn_hiddens, num_layers, max_len, dropout, **kwargs):
        super(BertEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embedding = BertEmbedding(vocab_size=vocab_size, embed_dim=embed_dim,
                                       max_len=max_len)
        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module(name=f'BertEncoderBlock{i}',
                                   module=EncoderBlock(embed_dim=embed_dim, num_heads=num_heads,
                                                       ffn_hiddens=ffn_hiddens, dropout=dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, segments, mask):
        X = self.embedding(tokens, segments)  # X shape: (N, max_len, embed_dim)
        for layer in self.layers:
            X = layer(X, X, X, mask)
        return X
