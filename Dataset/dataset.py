from torch.utils.data import Dataset
from utils import tokenize, Vocab, get_nsp_data, get_mlm_data, pad_bert_inputs


class TextDataset(Dataset):
    def __init__(self, paragraphs, max_len):
        # 三重列表的嵌套[[[]]], 段落、句子、词元
        paragraphs = [tokenize(lines=paragraph, token='word') for paragraph in paragraphs]
        # 两重列表的嵌套[[]], 句子、词元
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = Vocab(tokens=sentences, min_freq=1,
                           reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        examples = []
        # 获取下一句子预测任务的数据
        for paragraph in paragraphs:
            examples.extend(get_nsp_data(paragraph, paragraphs, max_len))
        # 获取遮蔽语言模型任务的数据
        # examples[(input_tokens_ids, pred_positions, mlm_pred_labels_ids, segments, is_next), ..., ()]
        examples = [(get_mlm_data(tokens, self.vocab) + (segments, is_next))
                    for tokens, segments, is_next in examples]

        # all_token_ids: 列表嵌套张量, tensor的shape为[64], 输入的tokens
        # all_segments: 列表嵌套张量, tensor的shape为[64], segments
        # valid_lens: 列表嵌套张量, tensor的shape为[1], 真实长度
        # all_pred_positions: 列表嵌套张量, tensor的shape为[10]
        # all_mlm_weights: 列表嵌套张量, tensor的shape为[10], 用于计算损失, 其中1的位置表示需要计算损失的位置(mask的位置)
        # all_mlm_labels: 列表嵌套张量, tensor的shape为[10], mlm需要预测的tokens的labels
        # nsp_labels: 列表嵌套张量, tensor的shape为[1], nsp任务的label
        (self.all_token_ids, self.all_segments, self.valid_lens, self.all_pred_positions,
         self.all_mlm_weights, self.all_mlm_labels, self.nsp_labels) = pad_bert_inputs(examples=examples,
                                                                                       max_len=max_len,
                                                                                       vocab=self.vocab)

    def __len__(self):
        return len(self.all_token_ids)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx],
                self.all_pred_positions[idx], self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])
