from torch.utils.data import DataLoader
from utils import read_data
from .dataset import TextDataset


def load_data(data_dir, batch_size, max_len, mode='train'):
    """加载wikiText-2数据集"""
    file = 'wiki.' + mode + '.tokens'
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    paragraphs = read_data(data_dir=data_dir, file=file)  # 读取整个文本, 两重列表的嵌套[[]]
    train_dataset = TextDataset(paragraphs, max_len)
    train_iter = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return train_iter, train_dataset.vocab
