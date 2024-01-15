from tqdm import tqdm
import logging
from utils import grad_clipping


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    model = model.to(device)
    model.train()

    total_loss = 0.
    mlm_accuracy = 0.
    nsp_accuracy = 0.
    iterations = len(dataloader)

    # 创建进度条
    pbar = tqdm(desc=f"epoch: {epoch + 1}", total=iterations, postfix=dict, mininterval=0.4)
    for iteration, batch_iter in enumerate(dataloader):
        tokens, segments, valid_lens, pred_positions, mlm_weights, mlm_target, nsp_target = [X.to(device) for X in
                                                                                             batch_iter]
        _, mlm_pred, nsp_pred = model(tokens, segments, valid_lens, pred_positions)
        optimizer.zero_grad()
        loss = loss_fn(mlm_pred, nsp_pred, mlm_target, nsp_target, mlm_weights)
        loss.backward()
        grad_clipping(model, 1)
        optimizer.step()
        # total_loss 是epoch中的每个batch的平均loss的总和
        total_loss += loss.item()
        # 计算mlm任务的准确率
        mlm_acc = (mlm_pred.argmax(dim=-1) == mlm_target).sum().item() / mlm_target.numel()
        mlm_accuracy += mlm_acc
        # 计算nsp任务的准确率
        nsp_acc = (nsp_pred.argmax(dim=-1) == nsp_target).sum().item() / nsp_target.numel()
        nsp_accuracy += nsp_acc
        # 记录训练日志
        logging.info(f"Epoch: {epoch + 1}, Batch: {iterations * epoch + iteration + 1}, Loss: {loss.item()},"
                     f"mlm_Accuracy: {mlm_acc:.2%}, nsp_Accuracy: {nsp_acc:.2%}")

        pbar.set_postfix(**{'loss': f"{total_loss / (iteration + 1):.4f}",
                            'mlm_Accuracy': f"{mlm_accuracy / (iteration + 1):.2%}",
                            'nsp_Accuracy': f"{nsp_accuracy / (iteration + 1):.2%}",
                            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
                            'device': f"{tokens.device}"}
                         )
        pbar.update(1)
    pbar.close()
    checkpoint = {"epoch": epoch,
                  "model_state_dict": model.state_dict(),
                  "lr": optimizer.param_groups[0]['lr']}
    return checkpoint
