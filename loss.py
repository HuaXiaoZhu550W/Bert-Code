import torch.nn as nn


class MNLoss(nn.Module):
    def __init__(self, vocab_size, reduction='none', **kwargs):
        super(MNLoss, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, mlm_pred, nsp_pred, mlm_target, nsp_target, mlm_weights):
        # 计算mlm任务的损失
        mlm_loss = self.loss(mlm_pred.transpose(1, 2), mlm_target) * mlm_weights
        mlm_loss = mlm_loss.sum() / (mlm_weights.sum() + 1e-10)

        # 计算nsp任务的损失
        nsp_loss = self.loss(nsp_pred, nsp_target).mean()

        loss = mlm_loss + nsp_loss
        return loss
