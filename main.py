import os
import logging
import torch
import pickle
from Model import Bert
from Dataset import load_data
from loss import MNLoss
import torch.optim as optim
from train import train_epoch
from eval import evaluate
from utils import init_weights
from config import opt


def main(is_continue=False):
    train_loader, vocab = load_data(opt.data_dir, opt.batch_size, opt.max_len)
    # 保存vocab文件
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    test_loader, _ = load_data(opt.data_dir, opt.eval_batch, opt.max_len, mode='test')

    model = Bert(vocab_size=len(vocab), embed_dim=opt.embed_dim, num_heads=opt.num_heads, ffn_hiddens=opt.ffn_hiddens,
                 num_layers=opt.num_layers, max_len=opt.max_len)

    loss_fn = MNLoss(vocab_size=len(vocab))

    # 断点续训练
    if is_continue:
        # 加载上一个epoch训练结果
        checkpoint = torch.load(os.path.join(opt.weight_path, 'checkpoint.pth'))
        epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        epoch = 0
        lr = opt.lr
        model = init_weights(model)

    optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)

    # 配置日志文件记录器
    logging.basicConfig(filename=os.path.join(opt.logs_path, 'train.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    best_loss = 100
    best_mlm = 0.
    best_nsp = 0.
    # 开始训练
    for epoch in range(epoch, opt.epochs):
        checkpoint = train_epoch(model, train_loader, optimizer, loss_fn, device=opt.device[0], epoch=epoch)
        loss, mlm_accuracy, nsp_accuracy = evaluate(model, test_loader, loss_fn, device=opt.device[0])
        if loss <= best_loss and mlm_accuracy >= best_mlm and nsp_accuracy >= best_nsp:
            torch.save(checkpoint, os.path.join(opt.weight_path, 'checkpoint.pth'))
            best_loss, best_mlm, best_nsp = loss, mlm_accuracy, nsp_accuracy


if __name__ == "__main__":
    main(is_continue=False)
