import numpy as np

from sklearn.metrics import auc

def _enumerate_thresholds(rec_errors, n=1000):
    # maximum value of the anomaly score for all time steps in the validation data
    thresholds, step_size = [], abs(np.max(rec_errors) - np.min(rec_errors)) / n
    th = np.min(rec_errors)
    thresholds.append(th)

    print(f'Threshold Range: ({np.min(rec_errors)}, {np.max(rec_errors)}) with Step Size: {step_size}')

    for i in range(n):
        th = th + step_size
        thresholds.append(th)
    
    return thresholds

def _compute_anomaly_scores(x, rec_x, x_val=None, scoring='square_mean'):
    if scoring == 'absolute':
        return np.mean(np.abs(x - rec_x), axis=-1)
    elif scoring == 'square_mean':
        return np.mean(np.square(x - rec_x), axis=-1) # ref. S-RNNs
    elif scoring == 'square_median':
        return np.median(np.square(x - rec_x), axis=-1)
    elif scoring == 'probability':
        return None # ref. RAMED expect to fill in

def set_thresholds(x, rec_x, is_reconstructed=True, n=1000, scoring='square_mean'):
    rec_errors = _compute_anomaly_scores(x, rec_x, scoring) if is_reconstructed else rec_x
    if len(rec_errors.shape) > 2:
        if scoring.split('_')[1] == 'mean':
            rec_errors = np.mean(rec_errors, axis=0)
        else:
            rec_errors = np.median(rec_errors, axis=0)
            
    thresholds = _enumerate_thresholds(rec_errors, n)
    return thresholds

def evaluate(x, rec_x, labels, is_reconstructed=True, n=1000, scoring='square_mean', x_val=None):
    TP, TN, FP, FN = [], [], [], []
    precision, recall, f1, fpr = [], [], [], []

    rec_errors = _compute_anomaly_scores(x, rec_x, scoring) if is_reconstructed else rec_x
    if len(rec_errors.shape) > 2:
        if scoring.split('_')[1] == 'mean':
            rec_errors = np.mean(rec_errors, axis=0)
        else:
            rec_errors = np.median(rec_errors, axis=0)
            
    thresholds = _enumerate_thresholds(rec_errors, n)
    
    for th in thresholds: # for each threshold
        TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0
        for t in range(len(x)): # for each time window
            # if any part of the segment has an anomaly, we consider it as anomalous sequence
            true_anomalies, pred_anomalies = set(np.where(labels[t] == 1)[0]), set(np.where(rec_errors[t] > th)[0])

            if len(pred_anomalies) > 0 and len(pred_anomalies.intersection(true_anomalies)) > 0:
                # correct prediction (at least partial overlap with true anomalies)
                TP_t = TP_t + 1
            elif len(pred_anomalies) == 0 and len(true_anomalies) == 0:
                # correct rejection, no predicted anomaly on no true labels
                TN_t = TN_t + 1 
            elif len(pred_anomalies) > 0 and len(true_anomalies) == 0:
                # false alarm (i.e., predict anomalies on no true labels)
                FP_t = FP_t + 1
            elif len(pred_anomalies) == 0 and len(true_anomalies) > 0:
                # predict no anomaly when there is at least one true anomaly within the seq.
                FN_t = FN_t + 1
        
        TP.append(TP_t)
        TN.append(TN_t)
        FP.append(FP_t)
        FN.append(FN_t)
    
    for i in range(len(thresholds)):
        precision.append(TP[i] / (TP[i] + FP[i] + 1e-8))
        recall.append(TP[i] / (TP[i] + FN[i] + 1e-8)) # recall or true positive rate (TPR)
        fpr.append(FP[i] / (FP[i] + TN[i] + 1e-8))
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': auc(recall, precision),
        'roc_auc': auc(fpr, recall), # auc(fpr, tpr)
        'thresholds': thresholds,
        'rec_errors': rec_errors,
        'thresholds': thresholds
    }


def _simulate_thresholds(rec_errors, n, verbose):
    # maximum value of the anomaly score for all time steps in the test data
    thresholds, step_size = [], abs(np.max(rec_errors) - np.min(rec_errors)) / n
    th = np.min(rec_errors)
    thresholds.append(th)
    
    if verbose:
        print(f'Threshold Range: ({np.max(rec_errors)}, {np.min(rec_errors)}) with Step Size: {step_size}')
    for i in range(n):
        th = th + step_size
        thresholds.append(th)

    return thresholds


def compute_traditional_metrics(anomaly_scores, labels, n=1000, delta=0., verbose=False):
    TP, TN, FP, FN = [], [], [], []
    precision, recall, f1, fpr = [], [], [], []

    thresholds = _simulate_thresholds(anomaly_scores, n, verbose)

    for th in thresholds: # for each threshold
        TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0
        for t in range(len(anomaly_scores)): # for each sequence

            # if any part of the segment has an anomaly, we consider it as anomalous sequence
            true_anomalies, pred_anomalies = set(np.where(labels[t] == 1)[0]), set(np.where(anomaly_scores[t] > th)[0])

            if len(pred_anomalies) > 0 and len(pred_anomalies.intersection(true_anomalies)) > 0:
                # correct prediction (at least partial overlap with true anomalies)
                TP_t = TP_t + 1
            elif len(pred_anomalies) == 0 and len(true_anomalies) == 0:
                # correct rejection, no predicted anomaly on no true labels
                TN_t = TN_t + 1 
            elif len(pred_anomalies) > 0 and len(true_anomalies) == 0:
                # false alarm (i.e., predict anomalies on no true labels)
                FP_t = FP_t + 1
            elif len(pred_anomalies) == 0 and len(true_anomalies) > 0:
                # predict no anomaly when there is at least one true anomaly within the seq.
                FN_t = FN_t + 1

        TP.append(TP_t)
        TN.append(TN_t)
        FP.append(FP_t)
        FN.append(FN_t)

    for i in range(len(thresholds)):
        precision.append(TP[i] / (TP[i] + FP[i] + 1e-7))
        recall.append(TP[i] / (TP[i] + FN[i] + 1e-7)) # recall or true positive rate (TPR)
        fpr.append(FP[i] / (FP[i] + TN[i] + 1e-7))
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-7))

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': auc(recall, precision),
        'roc_auc': auc(fpr, recall), # auc(fpr, tpr)
        'thresholds': thresholds
    }