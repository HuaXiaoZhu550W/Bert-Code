import torch
import pickle
from Model import Bert
from utils import get_tokens_and_segments
from config import opt


def get_bert_embedding(model, vocab, tokens_a, tokens_b=None, device='cpu'):
    """ 用bert预训练表示token """
    model.to(device)
    tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=device).unsqueeze(0)
    segments = torch.tensor(segments, device=device).unsqueeze(0)
    valid_lens = torch.tensor(len(tokens), device=device).unsqueeze(0)
    encoded, _, _ = model(token_ids, segments, valid_lens)
    return encoded


if __name__ == "__main__":
    # 读取vocab文件
    with open('vocab.pkl', 'rb') as f:
        Vocab = pickle.load(f)
    bert = Bert(vocab_size=len(Vocab), embed_dim=opt.embed_dim, num_heads=opt.num_heads,
                ffn_hiddens=opt.ffn_hiddens, num_layers=opt.num_layers, max_len=opt.max_len)
    a = ['he', 'like', 'apple']
    bert_embedding = get_bert_embedding(bert, Vocab, a, device=opt.device[0])

