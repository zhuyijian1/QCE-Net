import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from utils import AverageMeter


def train_epoch(epoch, model, loss_fn, train_loader, optim, logger, device, args):
    model.train()
    preds_list = []
    labels_list = []

    losses = AverageMeter('loss', logger)
    # mse_losses = AverageMeter('mse', logger)
    # tri_losses = AverageMeter('tri', logger)

    for i, (video_feat, audio_feat, flow_feat, label) in enumerate(train_loader):
        video_feat = video_feat.to(device)  # (b, t, c)
        audio_feat = audio_feat.to(device)  # (b, t, c)
        flow_feat = flow_feat.to(device)  # (b, t, c)
        label = label.float().to(device)
        out = model(video_feat, audio_feat, flow_feat, [1, 1, 1])
        pred = out['output']
        gate_loss = out.get('gate_loss', torch.tensor(0.0, device=device))
        loss = loss_fn(pred, label, out['embed'], out['embed2'], out["recon_loss"], args)
        # label = targets_a
        optim.zero_grad(set_to_none=True)
        loss.backward()
        # pdb.set_trace()
        optim.step()

        losses.update(loss.item(), label.shape[0])
        # mse_losses.update(mse, label.shape[0])
        # tri_losses.update(tri, label.shape[0])

        preds_list.append(pred.detach().cpu().numpy())
        labels_list.append(label.detach().cpu().numpy())

    preds = np.concatenate(preds_list, axis=0) if len(preds_list) > 0 else np.array([])
    labels = np.concatenate(labels_list, axis=0) if len(labels_list) > 0 else np.array([])
    coef, _ = spearmanr(preds, labels)
    if logger is not None:
        logger.add_scalar('train coef', coef, epoch)
    avg_loss = losses.done(epoch)
    # mse_losses.done(epoch)
    # tri_losses.done(epoch)
    return avg_loss, coef
