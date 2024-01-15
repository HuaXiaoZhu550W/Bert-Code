"""
超参数文件
"""
import os
import argparse
from utils import try_all_gpus

parser = argparse.ArgumentParser()
parser.add_argument('-max_len', type=int, default=64)
parser.add_argument('-embed_dim', type=int, default=768)
parser.add_argument('-ffn_hiddens', type=int, default=3072)
parser.add_argument('-num_layers', type=int, default=12)
parser.add_argument('-num_heads', type=int, default=12)
parser.add_argument('-dropout', type=float, default=0.2)
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-beta1', type=float, default=0.9)
parser.add_argument('-beta2', type=float, default=0.98)
parser.add_argument('-eps', type=float, default=1e-9)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-eval_batch', type=int, default=64)
parser.add_argument('-epochs', type=int, default=1000)
parser.add_argument('-warmup_steps', type=int, default=4000)
parser.add_argument('-device', type=str, default=try_all_gpus())  # 返回一个list,包含多GPU, 单CPU则需要device[0]
parser.add_argument('-data_dir',
                    default="F:/code_space/NLP/Data/wikitext-2")
parser.add_argument('-weight_path', default="weight/")
parser.add_argument('-logs_path', default="logs/")
opt = parser.parse_args(args=[])


if not os.path.exists(opt.weight_path):
    os.mkdir(opt.weight_path)

if not os.path.exists(opt.logs_path):
    os.mkdir(opt.logs_path)
