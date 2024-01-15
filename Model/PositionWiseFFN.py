import torch.nn as nn


class PositionWiseFFN(nn.Module):
    """
    前馈神经网络
    输入 ffn_inputs 就是H(base 768, large 1024)
    中间层 ffn_hiddens 是4*H
    输出 ffn_outputs 是H
    """

    def __init__(self, ffn_inputs, ffn_hiddens, ffn_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(in_features=ffn_inputs, out_features=ffn_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(in_features=ffn_hiddens, out_features=ffn_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
