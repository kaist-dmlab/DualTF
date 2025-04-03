import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

from sklearn.metrics import auc
from sklearn.preprocessing import RobustScaler
from utils.data_loader import load_tods, load_PSM, load_SWaT, load_ASD, load_CompanyA, load_ECG
from utils.data_loader import normalization, _create_sequences

def _simulate_thresholds(rec_errors, n):
    # maximum value of the anomaly score for all time steps in the test data
    thresholds, step_size = [], abs(np.max(rec_errors) - np.min(rec_errors)) / n
    th = np.min(rec_errors)
    thresholds.append(th)
    
    print(f'Threshold Range: ({np.min(rec_errors)}, {np.max(rec_errors)}) with Step Size: {step_size}')
    for i in range(n):
        th = th + step_size
        thresholds.append(th)

    return thresholds


parser = argparse.ArgumentParser(description='Settings for Daul-TF')
# Settings
parser.add_argument('--thresh_num', type=int, default=1000)
parser.add_argument('--seq_length', type=int, default=50)
parser.add_argument('--nest_length', type=int, default=10)
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--dataset', type=str, default='NeurIPSTS')
parser.add_argument('--form', type=str, default='seasonal')
parser.add_argument('--data_loader', type=str, default='load_tods')
parser.add_argument('--data_num', type=int, default=0)
opts = parser.parse_args()

# check settings
if opts.dataset == 'NeurIPSTS':
    print(f"Dataset: {opts.dataset}\nForm: {opts.form}\nSeq_length: {opts.seq_length}\nNest_length: {opts.nest_length}")
else:
    print(f"Dataset: {opts.dataset}\nNum: {opts.data_num}\nSeq_length: {opts.seq_length}\nNest_length: {opts.nest_length}")    

def total_evaluation():
    # Time array
    # df.index = ['Normal', 'Anomaly', '#Seq', 'Pred(%)', 'Pred', 'GT', 'Avg(RE)']
    print('Time Arrays Loading...')
    if opts.dataset == 'NeurIPSTS':
        file_path = f'./time_arrays/{opts.dataset}_{opts.form}_time_evaluation_array.pkl'
    else:
        file_path = f'./time_arrays/{opts.dataset}_{opts.data_num}_time_evaluation_array.pkl'
    time_array = pd.read_pickle(file_path)
    print(time_array.shape)
    print(time_array)

    # Frequency array
    # df.index = ['#SubSeq', '#GrandSeq', 'Avg(exp(RE))', 'Pred', 'GT']
    print('Frequency Arrays Loading...')
    if opts.dataset == 'NeurIPSTS':
        file_path = f'./freq_arrays/{opts.dataset}_{opts.form}_freq_evaluation_array.pkl'
    else:
        file_path = f'./freq_arrays/{opts.dataset}_{opts.data_num}_freq_evaluation_array.pkl'
    freq_array = pd.read_pickle(file_path)
    print(freq_array.shape)
    print(freq_array)

    # Summation
    time_rec = np.array(time_array.loc['Avg(RE)', :])
    freq_rec = np.array(freq_array.loc['Avg(exp(RE))', :])

    time_rec = time_rec.reshape(-1, 1)
    freq_rec = freq_rec.reshape(-1, 1)

    scaler = RobustScaler(unit_variance=True)
    time_as = scaler.fit_transform(time_rec)
    freq_as = scaler.transform(freq_rec)    

    time_as = normalization(time_rec)
    freq_as = normalization(freq_rec)

    final_as = time_as + freq_as

    # Point Adjusted Evaluation
    pa_scores = {'dataset': [], 'f1': [], 'precision': [], 'recall': [], 'pr_auc': [], 'roc_auc': []}
    print('##### Point Adjusted Evaluation #####')    
    thresholds = _simulate_thresholds(final_as, opts.thresh_num)
    final_as_seq = _create_sequences(final_as, opts.seq_length, opts.step)

    if opts.dataset == 'NeurIPSTS':
        label = globals()[f'{opts.data_loader}'](opts.form)['label_seq']
    else:
        label = globals()[f'{opts.data_loader}']()['label_seq'][opts.data_num]
    labels = _create_sequences(label, opts.seq_length, opts.step)

    TP, TN, FP, FN = [], [], [], []
    precision, recall, f1, fpr = [], [], [], []
    for th in tqdm(thresholds): # for each threshold
        TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0
        for t in range(len(final_as_seq)): # for each sequence

            # if any part of the segment has an anomaly, we consider it as anomalous sequence
            true_anomalies, pred_anomalies = set(np.where(labels[t] == 1)[0]), set(np.where(final_as_seq[t] > th)[0])

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

    highest_th_idx = np.argmax(f1)
    print(f'Threshold: {thresholds[highest_th_idx]}')
    print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            precision[highest_th_idx], recall[highest_th_idx], f1[highest_th_idx]))
    print("PR-AUC : {:0.4f}, ROC-AUC : {:0.4f}".format(auc(recall, precision), auc(fpr, recall)))
    
    pa_scores['dataset'].append(f'ECG{opts.data_num+1}')
    pa_scores['f1'].append(f1[highest_th_idx])
    pa_scores['precision'].append(precision[highest_th_idx])
    pa_scores['recall'].append(recall[highest_th_idx])
    pa_scores['pr_auc'].append(auc(recall, precision))
    pa_scores['roc_auc'].append(auc(fpr, recall))
    results = pd.DataFrame(pa_scores)
    print(results.groupby('dataset').mean())
    
    # Point-Wise Evaluation
    pw_scores = {'dataset': [], 'f1': [], 'precision': [], 'recall': [], 'pr_auc': [], 'roc_auc': []}
    print('##### Point-Wise Evaluation #####')
    TP, TN, FP, FN = [], [], [], []
    precision, recall, f1, fpr = [], [], [], []
    for th in tqdm(thresholds): # for each threshold    

        TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0
        for ts in range(len(final_as)):
            if label[ts] == 1:
                if final_as[ts] >= th:
                    TP_t = TP_t + 1
                elif final_as[ts] < th:
                    FN_t = FN_t + 1
            elif label[ts] == 0:
                if final_as[ts] >= th:
                    FP_t = FP_t + 1
                elif final_as[ts] < th:
                    TN_t = TN_t + 1

        TP.append(TP_t)
        TN.append(TN_t)
        FP.append(FP_t)
        FN.append(FN_t)
            
    for i in range(len(thresholds)):
        precision.append(TP[i] / (TP[i] + FP[i] + 1e-8))
        recall.append(TP[i] / (TP[i] + FN[i] + 1e-8)) # recall or true positive rate (TPR)
        fpr.append(FP[i] / (FP[i] + TN[i] + 1e-8))
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8))
    
    highest_th_idx = np.argmax(f1)
    print(f'Threshold: {thresholds[highest_th_idx]}')
    print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            precision[highest_th_idx], recall[highest_th_idx], f1[highest_th_idx]))
    print("PR-AUC : {:0.4f}, ROC-AUC : {:0.4f}".format(auc(recall, precision), auc(fpr, recall)))

    pw_scores['dataset'].append(f'ECG{opts.data_num+1}')
    pw_scores['f1'].append(f1[highest_th_idx])
    pw_scores['precision'].append(precision[highest_th_idx])
    pw_scores['recall'].append(recall[highest_th_idx])
    pw_scores['pr_auc'].append(auc(recall, precision))
    pw_scores['roc_auc'].append(auc(fpr, recall))
    results = pd.DataFrame(pw_scores)
    print(results.groupby('dataset').mean())      


    # Released Point-Wise Evaluation
    print('##### Released Point-Wise Evaluation #####')
    pr_scores = {'dataset': [], 'f1': [], 'precision': [], 'recall': [], 'pr_auc': [], 'roc_auc': []}
    thresholds = _simulate_thresholds(final_as, opts.thresh_num)
    final_as_seq = _create_sequences(final_as, opts.nest_length, opts.step)    
    labels = _create_sequences(label, opts.nest_length, opts.step)

    TP, TN, FP, FN = [], [], [], []
    precision, recall, f1, fpr = [], [], [], []
    for th in tqdm(thresholds): # for each threshold
        TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0
        for t in range(len(final_as_seq)): # for each sequence

            # if any part of the segment has an anomaly, we consider it as anomalous sequence
            true_anomalies, pred_anomalies = set(np.where(labels[t] == 1)[0]), set(np.where(final_as_seq[t] > th)[0])

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

    highest_th_idx = np.argmax(f1)
    print(f'Threshold: {thresholds[highest_th_idx]}')
    print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            precision[highest_th_idx], recall[highest_th_idx], f1[highest_th_idx]))
    print("PR-AUC : {:0.4f}, ROC-AUC : {:0.4f}".format(auc(recall, precision), auc(fpr, recall)))

    pr_scores['dataset'].append(f'ECG{opts.data_num+1}')
    pr_scores['f1'].append(f1[highest_th_idx])
    pr_scores['precision'].append(precision[highest_th_idx])
    pr_scores['recall'].append(recall[highest_th_idx])
    pr_scores['pr_auc'].append(auc(recall, precision))
    pr_scores['roc_auc'].append(auc(fpr, recall))   
    results = pd.DataFrame(pr_scores)
    print(results.groupby('dataset').mean())     


    # Point-Wise Evaluation V2
    print('##### Released Point-Wise Evaluation V2 #####')
    prv2_scores = {'dataset': [], 'f1': [], 'precision': [], 'recall': [], 'pr_auc': [], 'roc_auc': []}
    thresholds = _simulate_thresholds(final_as, opts.thresh_num)
    final_as_seq = _create_sequences(final_as, opts.nest_length, opts.nest_length)
    labels = _create_sequences(label, opts.nest_length, opts.nest_length)

    TP, TN, FP, FN = [], [], [], []
    precision, recall, f1, fpr = [], [], [], []
    for th in tqdm(thresholds): # for each threshold
        TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0
        for t in range(len(final_as_seq)): # for each sequence

            # if any part of the segment has an anomaly, we consider it as anomalous sequence
            true_anomalies, pred_anomalies = set(np.where(labels[t] == 1)[0]), set(np.where(final_as_seq[t] > th)[0])

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

    highest_th_idx = np.argmax(f1)
    print(f'Threshold: {thresholds[highest_th_idx]}')
    print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            precision[highest_th_idx], recall[highest_th_idx], f1[highest_th_idx]))
    print("PR-AUC : {:0.4f}, ROC-AUC : {:0.4f}".format(auc(recall, precision), auc(fpr, recall)))

    prv2_scores['dataset'].append(f'ECG{opts.data_num+1}')
    prv2_scores['f1'].append(f1[highest_th_idx])
    prv2_scores['precision'].append(precision[highest_th_idx])
    prv2_scores['recall'].append(recall[highest_th_idx])
    prv2_scores['pr_auc'].append(auc(recall, precision))
    prv2_scores['roc_auc'].append(auc(fpr, recall))   
    results = pd.DataFrame(prv2_scores)
    print(results.groupby('dataset').mean())

if __name__ == '__main__':
    total_evaluation()
