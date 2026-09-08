import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn


def min_max_normalize_along_axis(array, axis):
    min_vals = np.min(array, axis=axis, keepdims=True)
    max_vals = np.max(array, axis=axis, keepdims=True)
    normalized_array = (array - min_vals) / (max_vals - min_vals)
    return normalized_array


def test_epoch(epoch, model, test_loader, logger, device, mask, args):
    mse_loss = nn.MSELoss().to(device)
    model.eval()
    preds_list = []
    labels_list = []
    tol_loss, tol_sample = 0, 0

    feats = []

    with torch.no_grad():
        for i, (video_feat, audio_feat, flow_feat, label) in enumerate(test_loader):
            video_feat = video_feat.to(device)
            audio_feat = audio_feat.to(device)  # (b, t, c)
            flow_feat = flow_feat.to(device)  # (b, t, c)
            label = label.float().to(device)
            out = model(video_feat, audio_feat, flow_feat, mask)
            pred = out['output']

            if 'encode' in out.keys() and out['encode'] is not None:
                feats.append(out['encode'].mean(dim=1).cpu().detach().numpy())
                # feats.append(out['embed'].cpu().detach().numpy())

            score_range = args.score_range if args.score_range is not None else 1.0
            loss = mse_loss(pred * score_range, label * score_range)
            tol_loss += (loss.item() * label.shape[0])
            tol_sample += label.shape[0]

            preds_list.append(pred.detach().cpu().numpy())
            labels_list.append(label.detach().cpu().numpy())
    # print(preds)
    preds = np.concatenate(preds_list, axis=0) if len(preds_list) > 0 else np.array([])
    labels = np.concatenate(labels_list, axis=0) if len(labels_list) > 0 else np.array([])
    avg_coef, _ = spearmanr(preds, labels)
    avg_loss = float(tol_loss) / float(tol_sample)

    if logger is not None:
        logger.add_scalar('Test coef', avg_coef, epoch)
        logger.add_scalar('Test loss', avg_loss, epoch)
    # print(preds.tolist())
    # print(labels.tolist())
    return avg_loss, avg_coef
