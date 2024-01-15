import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """输入 embed_dim(base 768, large 1024), num_heads(base 12, large 16), head_dim 64 """
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == embed_dim), "Embedding dim needs to be divisible by heads"

        self.query = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.key = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.value = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.out_layer = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, queries, keys, values, mask):
        queries = self.split(self.query(queries))
        keys = self.split(self.key(keys))
        values = self.split(self.value(values))

        # queries*keys -> energy shape: (N, num_heads, query_len, key_len)
        energy = torch.matmul(queries, keys.transpose(2, 3)) / self.head_dim ** 0.5

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e10'))

        # 计算attention
        attention = torch.softmax(energy, dim=-1)

        # attention * values -> att_out shape: (N, num_heads, query_len, head_dim)
        att_out = torch.matmul(attention, values)

        return self.out_layer(self.concat(att_out))

    def split(self, X):
        # X shape: (N, max_lens, embed_dim), split shape: (N, num_heads, max_lens, head_dim)
        X = X.view(X.shape[0], X.shape[1], self.num_heads, self.head_dim)
        return X.transpose(1, 2)

    def concat(self, X):
        # X shape: (N, num_heads, query_len, head_dim), concat shape: (N, query_len, embed_dim)
        X = X.transpose(1, 2)
        return X.reshape(X.shape[0], X.shape[1], self.embed_dim)
