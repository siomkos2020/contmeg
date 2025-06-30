
import math
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def get_seq_generation_metrics(type_preds, time_preds, labels, time_gt, pred_probs, END_ID):
    r"""Compute the metrics for evaluating sequence generation."""
    k = len(type_preds[0])
    lls = [0] * k
    accs = [0] * k                                                  # Accuracy at future K points.
    rmses = [0] * k                                                 # RMSE at future K points.
    aucs = [0] * k
    tens_pred_probs = torch.tensor(pred_probs)
    tens_labels = torch.zeros_like(tens_pred_probs)
    for i in range(len(labels)):
        seq_len = len(labels[i])
        for j in range(min(len(labels[i]), k)):
            if j < seq_len:
                tens_labels[i][j][labels[i][j]] = 1

    for i in range(k):
        aucs[i] = roc_auc_score(torch.flatten(tens_labels[:, :i+1]).long().numpy(),
                                torch.flatten(tens_pred_probs[:, :i+1]).numpy())
    # Compute follow-up performances.
    # Decode the predictions into sequences and then calculate metrics.
    pred_seqs, gt_seqs, pred_times, gt_times, type_probs = [], [], [], [], []
    for i in range(len(type_preds)):
        type_seq_i, time_seq_i = [], []
        gt_type_seq_i, gt_time_seq_i = [], []
        type_prob_i = []
        seq_len = len(labels[i])
        # Select the first k predicted points.
        for j in range(min(seq_len, k)):
            if j < seq_len and labels[i][j] != END_ID:
                type_seq_i.append(type_preds[i][j])
                time_seq_i.append(time_preds[i][j])
                gt_type_seq_i.append(labels[i][j])
                gt_time_seq_i.append(time_gt[i][j])
                type_prob_i.append(pred_probs[i][j][labels[i][j]])
            else: break

        pred_seqs.append(type_seq_i)
        gt_seqs.append(gt_type_seq_i)
        pred_times.append(time_seq_i)
        gt_times.append(gt_time_seq_i)
        type_probs.append(type_prob_i)
    
    # Compute metrics by different time points.
    for i in range(k):
        ll, total_ll, correct_types, total_types, rmse_time, total_rmse = 0, 0, 0, 0, 0, 0
        for j in range(len(pred_seqs)):
            correct_types += 1 if tuple(pred_seqs[j][:i+1]) == tuple(gt_seqs[j][:i+1]) else 0
            total_types += 1   
            # Compute multi-label auc
            rmse_time += np.sqrt(((np.array(pred_times[j][:i+1]) - np.array(gt_times[j][:i+1]))**2).mean())
            total_rmse += 1
            ll += sum([math.log(x+1e-6) for x in type_probs[j][:i+1]])
            total_ll += 1
        lls[i] = round(ll / total_ll, 4)
        accs[i] = round(correct_types / total_types, 4)
        rmses[i] = round(rmse_time / total_rmse, 4)

    return lls, rmses, accs, aucs 

