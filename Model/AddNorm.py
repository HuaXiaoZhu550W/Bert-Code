import torch.nn as nn


class AddNorm(nn.Module):
    """残差链接和层归一化"""
    def __init__(self, normalized_shape, dropout):
        """
        :param normalized_shape: [seq_len, embed_dim]
        :param dropout: 0.0 ~ 1.0
        """
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
