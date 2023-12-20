import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt


def compute_window_packet_frequency_threshold(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    upper = np.percentile(data, 75) + 1.5 * iqr
    lower = np.percentile(data, 25) - 1.5 * iqr
    if lower < 0:
        lower = 0.0
    return lower, upper


def compute_windows(packet_data, timestamp_data):
    window_packet_frequencies= []
    for i in range(len(timestamp_data)):
        window_packet_data = packet_data[i]
        for j in range(i + 1, len(timestamp_data)):
            if timestamp_data[j] - timestamp_data[i] <= 10:
                window_packet_data += packet_data[j]
            else:
                break
        
        window_packet_frequencies.append(window_packet_data)
    return window_packet_frequencies

def compute_window_labels(timestamp_data, label_data):
    window_labels = []
    for i in range(len(timestamp_data)):
        labels =  set()
        labels.add(label_data[i])
        for j in range(i + 1, len(timestamp_data)):
            if timestamp_data[j] - timestamp_data[i] <= 10:
                labels.add(label_data[j])
            else:
                break
        
        if 'malicious' in labels:
            window_labels.append('malicious')
        else:
            window_labels.append('benign')

    return window_labels


def main():
    datasets = ['elastic', 'ctu', 'ugr']
    output_folder = 'boxplot_baseline_comparison'
    input_folder = 'data/'
    progress = tqdm(total= len(datasets) * 10, desc='Progress')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for dataset in datasets:
        train_data = pd.read_csv(input_folder + dataset + '_train_data_ENCODE.csv', low_memory=False)
        test_data = pd.read_csv(input_folder  + dataset + '_test_data_ENCODE.csv', low_memory=False)
        auc_scores = []
        train_time = []

        for i in range(10):
            start_time = time.time()
            train_windows_packets_data = compute_windows(train_data['num_packets'].tolist(), train_data['timestamp'].tolist())
            lower_threshold, upper_threshold = compute_window_packet_frequency_threshold(train_windows_packets_data)
            end_time = time.time()
            train_time.append(end_time - start_time)

            test_windows_packets_data = compute_windows(test_data['num_packets'].tolist(), test_data['timestamp'].tolist())
            actual_test_windows_labels = compute_window_labels(test_data['timestamp'].tolist(), test_data['type:label'].tolist())
            actual_test_windows_labels = [0 if x == 'benign' else 1 for x in actual_test_windows_labels]

            predicted_labels = []
            for i in range(len(test_windows_packets_data)):
                if test_windows_packets_data[i] >= lower_threshold and test_windows_packets_data[i] <= upper_threshold:
                    predicted_labels.append(0)
                else:
                    predicted_labels.append(1)
    
            print('Computing AUC...')
            y_true = actual_test_windows_labels
            y_pred = predicted_labels
            auc = roc_auc_score(y_true, y_pred)
            auc_scores.append(auc)
            progress.update(1)
        
        print('Writing average results to file...')
        with open(output_folder + '/' + dataset + '_boxplot_baseline_auc_score_10_runs.csv', 'w') as f:
            f.write('avg_auc_score,std_auc_score\n')
            f.write(str(np.mean(auc_scores)) + ',' + str(np.std(auc_scores)) + '\n')
        
        with open(output_folder + '/' + dataset + '_boxplot_baseline_train_time_10_runs.csv', 'w') as f:
            f.write('avg_train_time,std_train_time\n')
            f.write(str(np.mean(train_time)) + ',' + str(np.std(train_time)) + '\n')



if __name__ == '__main__':
    main()