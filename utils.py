import os
import random
import torch
import collections
import torch.nn as nn


def read_data(data_dir, file):
    """
    读取数据
    :param data_dir: 数据文件所在目录
    :param file: 文件名
    """
    file_name = os.path.join(data_dir, file)
    with open(file_name, 'r', encoding='utf8') as f:
        lines = f.readlines()
    paragraphs = [line.strip().lower().split(' . ') for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def tokenize(lines, token='word'):
    """词元化(单词或者字符)"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print(f"Error: unknown token type: {token}")


class Vocab:
    """文本词汇表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计词频
        counter = self.count_corpus(tokens)
        # 按照词频排序
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens  # ['<unk>', '<pad>', '<mask>', '<cls>', '<sep>']
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}  # {'<unk>': 0, ...}
        # 过滤掉频率低的词
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """返回tokens和对应的index, {token: index}"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """返回index对应的token"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

    @staticmethod
    def count_corpus(tokens):
        """统计词元的频率, 这里的tokens是1D列表或2D列表"""
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)  # 返回一个字典 {'the': 129839, ...}


def get_next_sentence(sentence, next_sentence, paragraphs):
    """
    生成二分类任务的训练样本
    paragraphs是三重列表的嵌套[[[]]]
    """
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs))  # 随机抽取一个sentence
        is_next = False
    return sentence, next_sentence, is_next


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """
    获取:
    tokens: ['<cls>', tokens_a, '<sep>', tokens_b, '<sep>']
    segments: [0, 0, ..., 0, 1, 1, ..., 1]
    """
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 表示tokens_a的token
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        # 1 表示tokens_b的token
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def get_nsp_data(paragraph, paragraphs, max_len):
    """
    nsp_data: [(tokens, segments, is_next), ...]
    """
    nsp_data = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = get_next_sentence(sentence=paragraph[i], next_sentence=paragraph[i + 1],
                                                        paragraphs=paragraphs)
        # 输入: <cls> + tokens_a + <seq> + tokens_b + <sep>, 因此要加上3
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data.append((tokens, segments, is_next))
    return nsp_data


def replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    """
    为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    """
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 将预测位置打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        # 80%的概率, 将词替换为'<mask>'
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的概率, 保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的概率, 用随机词替换该词
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def get_mlm_data(tokens, vocab):
    """
    # tokens是一个字符串列表
    """
    candidate_pred_positions = []  # 候选预测位置
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)

    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))  # int
    mlm_input_tokens, pred_positions_and_labels = replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def pad_bert_inputs(examples, max_len, vocab):
    """
    对inputs进行填充操作
    """
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids = []
    all_segments = []
    valid_lens = []
    all_pred_positions = []  # 需要预测的位置
    all_mlm_weights = []
    all_mlm_labels = []  # 遮蔽语言模型的label
    nsp_labels = []  # 预测下一句子的label
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        # 用'<pad>'(1)填充, 每个tensor的长度为max_len
        all_token_ids.append(torch.tensor(token_ids +
                                          [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
        # 用0填充, 每个tensor的长度为max_len
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        # 用0填充, 每个tensor的长度为max_len
        all_pred_positions.append(torch.tensor(pred_positions +
                                               [0] * (max_num_mlm_preds - len(pred_positions)),
                                               dtype=torch.long))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) +
                                            [0.0] * (max_num_mlm_preds - len(pred_positions)),
                                            dtype=torch.float32))
        # 用0填充, 每个tensor的长度为10
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids +
                                           [0.0] * (max_num_mlm_preds - len(mlm_pred_label_ids)),
                                           dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


def make_mask(lens, max_len):
    """
    生成填充掩码
    lens: shape (batch_size,)
    max_length: int
    pad_mask: shape (batch_size, 1, max_length, max_length)
    """
    pad = torch.arange(max_len, device=lens.device)[None, :] < lens[:, None]
    pad = pad.unsqueeze(2).to(torch.float32)
    pad_mask = torch.bmm(pad, pad.transpose(1, 2))
    return pad_mask.unsqueeze(1)


def try_all_gpus():
    """ 获取所用gpu """
    device = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return device if device else [torch.device('cpu')]


def grad_clipping(net, theta):
    """梯度剪裁"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params if p.grad is not None))
    if norm > theta:
        for param in params:
            if param.grad is not None:
                param.grad[:] *= theta / norm


def init_weights(model):
    for name, param in model.named_parameters():
        if param.dim() > 1:
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, val=0.)
            else:
                nn.init.xavier_uniform_(param)
    return model
