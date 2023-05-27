import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

import os
import time

from model.TimeTransformer import AnomalyTransformer
from model.FrequencyTransformer import FrequencyTransformer
from utils.data_loader import get_loader_segment, get_loader_grandwin


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class TimeReconstructor(object):
    DEFAULTS = {}

    def __init__(self, opts):

        self.__dict__.update(TimeReconstructor.DEFAULTS, **opts)

        self.build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        
    def build_model(self):
        # source: https://github.com/thuml/Anomaly-Transformer
        self.model = AnomalyTransformer(win_size=self.seq_length, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()
        
    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)).detach())) + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)).detach(), series[u])))
                prior_loss += (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)), series[u].detach())) + torch.mean(my_kl_loss(series[u].detach(),(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)


    def progress(self):
        print("======================TRAIN MODE======================")
        
        time_now = time.time()
        if self.dataset == 'NeurIPSTS':
            print(f'================={self.dataset}_{self.form}======================')
            train_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size, seq_length=self.seq_length, form=self.form, step=self.step, mode='train', dataset=self.dataset)
            vali_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size, seq_length=self.seq_length, form=self.form, step=self.step, mode='vali', dataset=self.dataset)
            test_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size, seq_length=self.seq_length, form=self.form, step=self.step, mode='test', dataset=self.dataset)
            thre_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size, seq_length=self.seq_length, form=self.form, step=self.step, mode='thre', dataset=self.dataset)
        else:
            print(f'================={self.dataset}_{self.data_num}======================')
            train_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size, seq_length=self.seq_length, form=self.data_num, step=self.step, mode='train', dataset=self.dataset)
            vali_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size, seq_length=self.seq_length, form=self.data_num, step=self.step, mode='vali', dataset=self.dataset)
            test_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size, seq_length=self.seq_length, form=self.data_num, step=self.step, mode='test', dataset=self.dataset)
            thre_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size, seq_length=self.seq_length, form=self.data_num, step=self.step, mode='thre', dataset=self.dataset)        
        
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            epoch_time = time.time()
            self.model.train()
                
            for i, input_data in enumerate(train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)).detach())) + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)).detach(), series[u])))
                    prior_loss += (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)),series[u].detach())) + torch.mean(my_kl_loss(series[u].detach(), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(test_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        print("Total Time: {}".format(time.time() - time_now))

        print("======================TEST MODE======================")
        if self.dataset == 'NeurIPSTS':
            print(f'================={self.dataset}_{self.form}======================')
        else:
            print(f'================={self.dataset}_{self.data_num}======================')
        self.model.load_state_dict(torch.load(os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        criterion = nn.MSELoss(reduction='none')

        # (1) stastic on the train set
        attens_energy = []
        for i, input_data in enumerate(train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)).detach()) * temperature
                    prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)),series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)).detach()) * temperature
                    prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)),series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)).detach()) * temperature
                    prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)), series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)).detach()) * temperature
                    prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)), series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        inference_time = time.time()
        for i, (input_data, labels) in enumerate(thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)).detach()) * temperature
                    prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)), series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)).detach()) * temperature
                    prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_length)), series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
        
        print(f'Test labels shape: {test_labels[0].shape}')
        attens_energy_old = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels_old = np.concatenate(test_labels, axis=0).reshape(-1)
        attens_energy_new = np.concatenate(attens_energy, axis=0)
        test_labels_new = np.concatenate(test_labels, axis=0)

        test_energy = np.array(attens_energy_old)
        test_labels = np.array(test_labels_old)
        test_energy_new = np.array(attens_energy_new)
        test_labels_new = np.array(test_labels_new)    

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)
        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        print("### Inference time: {}".format(time.time() - inference_time))

        print('########## New Evaluation ##########')
        evaluation_arrays = []
        # For plotting evaluation results
        evaluation_array = np.zeros((7, len(test_seq)))
        predicted_normal_array = np.zeros((len(test_seq)))
        predicted_anomaly_array = np.zeros((len(test_seq)))
        rec_error_array = np.zeros((len(test_seq)))

        num_context = 0
        for ts in range(len(test_seq)):
            if ts < self.seq_length - 1:
                num_context = ts + 1
            elif ts >= self.seq_length - 1 and ts < len(test_seq) - self.seq_length + 1:
                num_context = self.seq_length
            elif ts >= len(test_seq) - self.seq_length + 1:
                num_context = len(test_seq) - ts
            evaluation_array[2][ts] = num_context

        pred_anomal_idx = []
        # Per each window
        print(f'Energy shape: {test_energy_new.shape}')
        print(f'Energy median: {np.median(test_energy_new)}')
        threshold = np.median(test_energy_new)
        for t in range(len(test_energy_new)):
            # For reconstruction error sum
            rec_error_array[t:t+self.seq_length] += test_energy_new[t]

            pred_normals = np.where(test_energy_new[t] <= threshold)[0]
            pred_anomalies = np.where(test_energy_new[t] > threshold)[0]

            # For Noraml
            for j in range(len(pred_normals)):
                predicted_normal_array[pred_normals[j] + t] += 1
            # For Abnormal
            for k in range(len(pred_anomalies)):
                predicted_anomaly_array[pred_anomalies[k] + t] += 1

        evaluation_array[0] = predicted_normal_array
        evaluation_array[1] = predicted_anomaly_array
        
        # Reconstruction Errors
        evaluation_array[6] = rec_error_array/evaluation_array[2]

        # Predicted Anomaly Percentage
        for s in range(len(predicted_anomaly_array)):
            evaluation_array[3][s] = evaluation_array[1][s]/evaluation_array[2][s]

            # Predicted Anomaly (Binary)
            if evaluation_array[3][s] > 0.5:
                evaluation_array[4][s] = 1

        evaluation_array[5] = label_seq
        evaluation_arrays.append(evaluation_array)
        
        ## Evaluation Results
        eval_dfs=[]
        for i in range(len(evaluation_arrays)):
            print(f'Evaluation Array Shape: {evaluation_arrays[i].shape}')
            df = pd.DataFrame(evaluation_arrays[i])
            df.index = ['Normal', 'Anomaly', '#Seq', 'Pred(%)', 'Pred', 'GT', 'Avg(RE)']
            df = df.astype('float')
            eval_dfs.append(df)

        ## Save
        print(f'Saving Time Arrays... {self.dataset}')
        df_save_path = './time_arrays'
        if not os.path.exists(df_save_path):
            os.makedirs(df_save_path)
        for data_num in range(len(eval_dfs)):
            if self.dataset == 'NeurIPSTS':
                eval_dfs[data_num].to_pickle(f'{df_save_path}/{self.dataset}_{self.form}_time_evaluation_array.pkl')
            else:
                eval_dfs[data_num].to_pickle(f'{df_save_path}/{self.dataset}_{self.data_num}_time_evaluation_array.pkl')

        # Evaluation Metrics
        TP, TN, FP, FN = [], [], [], []
        precision, recall, f1, fpr = [], [], [], []
        for data_num in range(len(eval_dfs)):
            TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0
            for ts in eval_dfs[data_num].columns:
                if eval_dfs[data_num][ts]['GT'] == 1:
                    if eval_dfs[data_num][ts]['Pred'] == 1:
                        TP_t = TP_t + 1
                    elif eval_dfs[data_num][ts]['Pred'] == 0:
                        FN_t = FN_t + 1
                elif eval_dfs[data_num][ts]['GT'] == 0:
                    if eval_dfs[data_num][ts]['Pred'] == 1:
                        FP_t = FP_t + 1
                    elif eval_dfs[data_num][ts]['Pred'] == 0:
                        TN_t = TN_t + 1

            TP.append(TP_t)
            TN.append(TN_t)
            FP.append(FP_t)
            FN.append(FN_t)
                
        for i in range(len(TP)):
            precision.append(TP[i] / (TP[i] + FP[i] + 1e-8))
            recall.append(TP[i] / (TP[i] + FN[i] + 1e-8)) # recall or true positive rate (TPR)
            fpr.append(FP[i] / (FP[i] + TN[i] + 1e-8))
            f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8))
            # print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(precision[i], recall[i], f1[i]))

        print('########## Point Adjusted Evaluation ##########')
        # detection adjustment
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        # print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))

        return accuracy, precision, recall, f_score


class FreqReconstructor(object):
    DEFAULTS = {}

    def __init__(self, opts):

        self.__dict__.update(FreqReconstructor.DEFAULTS, **opts)

        self.build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = FrequencyTransformer(win_size=(self.seq_length-self.nest_length+1)*(self.nest_length//2), enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))).detach())) + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))).detach(), series[u])))
                prior_loss += (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))), series[u].detach())) + torch.mean(my_kl_loss(series[u].detach(),(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)  

    def progress(self):
        print("===========================TRAIN MODE===========================")
        if self.dataset == 'NeurIPSTS':
            print(f'================={self.dataset}_{self.form}======================')
            train_loader, test_loader, label_seq, test_seq, y_tests, grand_label = get_loader_grandwin(self.batch_size, self.seq_length, self.nest_length, self.form, self.step, dataset=self.dataset, data_loader=self.data_loader)
        else:
            print(f'================={self.dataset}_{self.data_num}======================')
            train_loader, test_loader, label_seq, test_seq, y_tests, grand_label = get_loader_grandwin(self.batch_size, self.seq_length, self.nest_length, self.data_num, self.step, dataset=self.dataset, data_loader=self.data_loader)
        
    
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
        
            for i, input_data in enumerate(train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)
                loss = torch.mean(self.criterion(input, output), dim=-1)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))).detach())) + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))).detach(), series[u])))
                    prior_loss += (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))),series[u].detach())) + torch.mean(my_kl_loss(series[u].detach(), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)
            vali_loss1, vali_loss2 = self.vali(test_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
        print("Total Time: {}".format(time.time() - time_now))

        print("======================TEST MODE======================")
        if self.dataset == 'NeurIPSTS':
            print(f'================={self.dataset}_{self.form}======================')
        else:
            print(f'================={self.dataset}_{self.data_num}======================')    
        self.model.load_state_dict(torch.load(os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        criterion = nn.MSELoss(reduction='none')

        # (1) stastic on the train set
        attens_energy = []
        for i, input_data in enumerate(train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))).detach()) * temperature
                    prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))),series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))).detach()) * temperature
                    prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))),series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)        

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(test_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))).detach()) * temperature
                    prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))), series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))).detach()) * temperature
                    prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))), series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set

        ############################ Alignment Module ############################
        grand_labels = []
        sub_evaluation_arrays = []
        inference_time = time.time()
        ############################ Alignment Module ############################
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(test_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))).detach()) * temperature
                    prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))), series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))).detach()) * temperature
                    prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, (self.seq_length-self.nest_length+1)*(self.nest_length//2))), series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
                
        print(f'Test labels shape: {test_labels[0].shape}')
        attens_energy = np.concatenate(attens_energy, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        
        print(f'Test labels shape: {test_labels.shape}')
        print(f'Test energy shape: {test_energy.shape}')
        # Before
        # test_energy: (301, 76*25/2, d), test_labels: (301, 76*25, ), grand_label: (301, 76, 25,)
        # After
        # nested_test_energy: (301, 76, 25/2, d), test_labels: (301, 76, 25, )

        print("### Inference time: {}".format(time.time() - inference_time))

        nested_test_energy = test_energy.reshape(test_energy.shape[0], self.seq_length-self.nest_length+1, self.nest_length//2)
        ###################################### Alignment ##############################################

        for outer in range(len(nested_test_energy)):
            # as_frequency: (76, 25/2)
            # as_frequency = normalization(np.power(np.exp(np.linalg.norm(nested_test_energy[outer], axis=1)), 2))
            as_frequency = np.power(np.exp(np.linalg.norm(nested_test_energy[outer], axis=1)), 2)
            # as_frequency = np.exp(np.linalg.norm(nested_test_energy[outer], axis=1))
            # Ablation
            # as_frequency = np.linalg.norm(nested_test_energy[outer], axis=1)
            sub_evaluation_array = np.zeros((4, self.seq_length))
            rec_error_array = np.zeros((self.seq_length))

            num_context = 0
            for ts in range(self.seq_length):
                if ts < self.nest_length - 1:
                    num_context = ts + 1
                elif ts >= self.nest_length - 1 and ts < self.seq_length - self.nest_length + 1:
                    num_context = self.nest_length
                elif ts >= self.seq_length - self.nest_length + 1:
                    num_context = self.seq_length - ts
                sub_evaluation_array[0][ts] = num_context # SubSeq

            pred_anomal_idx = []
            # Per each window
            # nested shape: (76, 25, 1)
            for t in range(len(nested_test_energy[outer])):
                rec_error_array[t:t + self.nest_length] += as_frequency[t]
                
            sub_evaluation_array[1] = rec_error_array/sub_evaluation_array[0] # exponential average (reconstruction error)
            
            # Predicted Anomaly Percentage
            threshold = np.percentile(sub_evaluation_array[1], 100 - self.anormly_ratio)
            for s in range(self.seq_length):
                # Predicted Anomaly (Binary)
                if sub_evaluation_array[1][s] > thresh:
                    sub_evaluation_array[2][s] = 1 # predicted label

            sub_evaluation_array[3] = np.squeeze(y_tests[outer])
            sub_evaluation_arrays.append(sub_evaluation_array)
        ###################################### Alignment ##############################################
        
        sub_evaluation_arrays = np.array(sub_evaluation_arrays)
        grand_evaluation_array = np.zeros((5, len(test_seq)))
        grand_rec_error_array = np.zeros((len(test_seq)))

        # Grand window array (301, 7, 100)
        for outer_win in range(len(sub_evaluation_arrays)):
            grand_evaluation_array[0][outer_win:outer_win + self.seq_length] += sub_evaluation_arrays[outer_win][0] # sub-seq

            # For reconstruction error sum
            grand_rec_error_array[outer_win:outer_win+self.seq_length] += sub_evaluation_arrays[outer_win][1]

        grand_context = 0
        # (400)
        for timestamp in range(len(test_seq)):
            if timestamp < self.seq_length - 1:
                grand_context = timestamp + 1
            elif timestamp >= self.seq_length - 1 and timestamp < len(test_seq) - self.seq_length + 1:
                grand_context = self.seq_length
            elif timestamp >= len(test_seq) - self.seq_length + 1:
                grand_context = len(test_seq) - timestamp
            grand_evaluation_array[1][timestamp] = grand_context # grand-seq

        grand_evaluation_array[2] = grand_rec_error_array/grand_evaluation_array[1] # average exponential reconstruction error
        for s in range(len(test_seq)):
            # Predicted Anomaly (Binary)
            if grand_evaluation_array[2][s] > np.mean(grand_evaluation_array[2]):
                grand_evaluation_array[3][s] = 1 # predicted label

        grand_evaluation_array[4] = label_seq # ground truth
        
        print(f'Grand Evaluation Array Shape: {grand_evaluation_array.shape}')
        
        ## Evaluation Results
        eval_dfs=[]
        df = pd.DataFrame(grand_evaluation_array)
        df.index = ['#SubSeq', '#GrandSeq', 'Avg(exp(RE))', 'Pred', 'GT']
        df = df.astype('float')
        eval_dfs.append(df)
        
        ## Save
        print(f'Saving Freq Arrays... {self.dataset}')
        df_save_path = './freq_arrays'
        if not os.path.exists(df_save_path):
            os.makedirs(df_save_path)
        for data_num in range(len(eval_dfs)):
            if self.dataset == 'NeurIPSTS':
                eval_dfs[data_num].to_pickle(f'{df_save_path}/{self.dataset}_{self.form}_freq_evaluation_array.pkl')
            else:
                eval_dfs[data_num].to_pickle(f'{df_save_path}/{self.dataset}_{self.data_num}_freq_evaluation_array.pkl')


        # Evaluation Metrics
        TP, TN, FP, FN = [], [], [], []
        precision, recall, f1, fpr = [], [], [], []
        for data_num in range(len(eval_dfs)):
            TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0
            for ts in eval_dfs[data_num].columns:
                if eval_dfs[data_num][ts]['GT'] == 1:
                    if eval_dfs[data_num][ts]['Pred'] == 1:
                        TP_t = TP_t + 1
                    elif eval_dfs[data_num][ts]['Pred'] == 0:
                        FN_t = FN_t + 1
                elif eval_dfs[data_num][ts]['GT'] == 0:
                    if eval_dfs[data_num][ts]['Pred'] == 1:
                        FP_t = FP_t + 1
                    elif eval_dfs[data_num][ts]['Pred'] == 0:
                        TN_t = TN_t + 1

            TP.append(TP_t)
            TN.append(TN_t)
            FP.append(FP_t)
            FN.append(FN_t)
                
        for i in range(len(TP)):
            precision.append(TP[i] / (TP[i] + FP[i] + 1e-8))
            recall.append(TP[i] / (TP[i] + FN[i] + 1e-8)) # recall or true positive rate (TPR)
            fpr.append(FP[i] / (FP[i] + TN[i] + 1e-8))
            f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8))
            # print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(precision[i], recall[i], f1[i]))

        return precision, recall, f1