import torch
import torch.nn as nn


class BertEmbedding(nn.Module):
    """输入: vocab_size, 输出: embed_dim(隐藏层大小H, base 768, large 1024)"""
    def __init__(self, vocab_size, embed_dim, max_len, **kwargs):
        super(BertEmbedding, self).__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.segment_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn((1, max_len, embed_dim)))  # 位置编码可以学习

    def forward(self, tokens, segments):
        # tokens shape: (N, max_len)
        # segments shape: (N, max_len)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.positional_encoding[:, :X.shape[1]]
        return X
