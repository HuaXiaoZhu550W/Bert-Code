import torch
import torch.nn as nn
from .BertEncoder import BertEncoder
from utils import make_mask


class MaskLM(nn.Module):
    """Bert的遮蔽语言模型任务"""
    def __init__(self, vocab_size, embed_dim=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=embed_dim),
                                 nn.ReLU(),
                                 nn.LayerNorm(normalized_shape=embed_dim),
                                 nn.Linear(in_features=embed_dim, out_features=vocab_size))

    def forward(self, X, pred_positions):
        # X shape: (N, max_len, embed_dim)
        # pred_positions shape: (N, max_len*0.15)  # (词汇)的15%需要预测
        num_pred = pred_positions.shape[1]  # max_len*0.15

        # 假设batch_size=2, num_pred=3
        # 那么batch_idx是np.array([0,0,0,1,1,1])
        pred_positions = pred_positions.view(-1)
        batch_idx = torch.arange(0, X.shape[0])
        batch_idx = torch.repeat_interleave(batch_idx, num_pred)

        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.view(X.shape[0], num_pred, -1)
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    """Bert的下一句预测任务"""
    def __init__(self, embed_dim=768, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=embed_dim),
                                 nn.Tanh(),
                                 nn.Linear(in_features=embed_dim, out_features=2))

    def forward(self, X):
        # X shape: (N, embed_dim)
        return self.mlp(X)


class Bert(nn.Module):
    """Bert模型"""
    def __init__(self, vocab_size, embed_dim=768, num_heads=12, ffn_hiddens=3072, num_layers=12,
                 max_len=64, dropout=0.2, **kwargs):
        super(Bert, self).__init__(**kwargs)
        self.encoder = BertEncoder(vocab_size, embed_dim, num_heads, ffn_hiddens, num_layers, max_len, dropout)

        # 用于Bert的掩蔽语言模型任务
        self.mlm = MaskLM(vocab_size=vocab_size, embed_dim=embed_dim)

        # 用于Bert的下一句预测任务
        self.nsp = NextSentencePred(embed_dim=embed_dim)

    def forward(self, tokens, segments, valid_lens, pred_positions=None):
        mask = make_mask(valid_lens, tokens.shape[1])  # 生成encoder_mask
        encoder_out = self.encoder(tokens, segments, mask)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(X=encoder_out, pred_positions=pred_positions)
        else:
            mlm_Y_hat = None

        # 0是"<cls>"标记的索引
        nsp_Y_hat = self.nsp(X=encoder_out[:, 0, :])
        return encoder_out, mlm_Y_hat, nsp_Y_hat
