import torch
from tqdm import tqdm


def evaluate(model, dataloader, loss_fn, device):
    model.to(device)
    model.eval()

    total_loss = 0.
    mlm_accuracy = 0.
    nsp_accuracy = 0.
    iterations = len(dataloader)

    # 创建进度条
    pbar = tqdm(desc=f"eval", total=iterations, postfix=dict, mininterval=0.4)
    for iteration, batch_iter in enumerate(dataloader):
        tokens, segments, valid_lens, pred_positions, mlm_weights, mlm_target, nsp_target = [X.to(device) for X in
                                                                                             batch_iter]
        with torch.no_grad():
            _, mlm_pred, nsp_pred = model(tokens, segments, valid_lens, pred_positions)
            loss = loss_fn(mlm_pred, nsp_pred, mlm_target, nsp_target, mlm_weights)

        # total_loss 是epoch中的每个batch的平均loss的总和
        total_loss += loss.item()
        # 计算mlm任务的准确率
        mlm_acc = (mlm_pred.argmax(dim=-1) == mlm_target).sum().item() / mlm_target.numel()
        mlm_accuracy += mlm_acc
        # 计算nsp任务的准确率
        nsp_acc = (nsp_pred.argmax(dim=-1) == nsp_target).sum().item() / nsp_target.numel()
        nsp_accuracy += nsp_acc

        pbar.set_postfix(**{'loss': f"{total_loss / (iteration + 1):.4f}",
                            'mlm_Accuracy': f"{mlm_accuracy / (iteration + 1):.2%}",
                            'nsp_Accuracy': f"{nsp_accuracy / (iteration + 1):.2%}",
                            'device': f"{tokens.device}"}
                         )
        pbar.update(1)
    pbar.close()
    return total_loss/iterations, mlm_accuracy/iterations, nsp_accuracy/iterations
