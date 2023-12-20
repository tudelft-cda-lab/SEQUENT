from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.abod import ABOD
from encode.encode import encode
from encode.matrix_operations import load_matrix
from encode.utils import compute_percentiles
from encode.utils import find_percentile
import time

def add_identity(axes, *line_args, **line_kwargs):
    # https://stackoverflow.com/a/28216751/15406859
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)

def tranform_symbols_for_frequency_encoding(encoded_data):
    transformed_symbol_data = []
    for element in encoded_data:
        if type(element) == str and element.startswith('i_'):
            number = int(element.split('i_')[1])
            if number < 0:
                transformed_symbol_data.append(number - 1000)
            else:
                transformed_symbol_data.append(- 1 * number)
        else:
            transformed_symbol_data.append(int(element))

    return transformed_symbol_data

def add_normal_dist_noise_to_feat_data(feature_data):
    noise = np.random.normal(0, 0.1, feature_data.shape)
    return feature_data + noise


def main():
    # print('Reading data...')
    datasets = ['elastic', 'ctu', 'ugr']
    model_type = 'FastABOD'
    input_folder = 'data/'
    encoding = 'ENCODE'
    progress_bar = tqdm(total= len(datasets) * 10, desc='Progress')
    for dataset in datasets:
        auc_scores = dict()
        train_time = dict()
        auc_scores[encoding] = []
        train_time[encoding] = []

        for i in range(10):
            training_file = input_folder + dataset.upper() + '_train_data_' + encoding + '_feature_vectors_conn.csv'
            test_file = input_folder + dataset.upper() + '_test_data_' + encoding + '_feature_vectors_conn.csv'

            train_data = pd.read_csv(training_file, low_memory=False)
            test_data = pd.read_csv(test_file, low_memory=False)
            
            columns_to_drop = ['timestamp', 'src_ip', 'dst_ip']
            train_data.drop(columns_to_drop, axis=1, inplace=True)
            test_data.drop(columns_to_drop, axis=1, inplace=True)


            y_true = test_data['label'].tolist()
            test_data.drop(['label'], axis=1, inplace=True)
            y_true = [0 if x == 'benign' else 1 for x in y_true]
            y_train = train_data['label'].tolist()
            y_train = [0 if x == 'benign' else 1 for x in y_train]
            train_data.drop(['label'], axis=1, inplace=True)

            # Use for ABOD
            if model_type == 'FastABOD':
                # add noise to each row in the dataframe
                train_data = train_data.apply(lambda x: add_normal_dist_noise_to_feat_data(x), axis=0)
                test_data = test_data.apply(lambda x: add_normal_dist_noise_to_feat_data(x), axis=0)
                
                model = ABOD(n_neighbors=5, contamination=0.01, method='fast')

            elif model_type == 'IF':
                model = IForest(n_estimators=5, max_samples=50 , contamination=0.01, n_jobs=-1, verbose=0)
            
            elif model_type == 'HBOS':
                    model = HBOS(n_bins=10, alpha=0.1)

            # print('Fitting model...')
            start_time = time.time()
            model.fit(train_data.values)
            end_time = time.time()


            # print('Computing AUC...')
            # Compute AUC
            y_score = model.decision_function(test_data.values).tolist()
            auc = roc_auc_score(y_true, y_score)
            auc_scores[encoding].append(auc)
            train_time[encoding].append(end_time - start_time)
            progress_bar.update(1)
        
        print('Writing average and standard deviation of AUC scores to file...')
        with open(input_folder + dataset.upper() + '_' + model_type + '_auc_score_10_runs_ENCODE_feature_vectors_conn.csv', 'w') as f:
            f.write('Encoding,AVG,STD\n')
            f.write('ENCODE vectors,' + str(np.mean(auc_scores[encoding])) + ',' + str(np.std(auc_scores[encoding])) + '\n')

        print('Writing average and standard deviation of training time to file...')
        with open(input_folder + dataset.upper() + '_' + model_type + '_train_time_ENCODE_feature_vectors_conn.csv', 'w') as f:
            f.write('Encoding,AVG,STD\n')
            f.write('ENCODE vectors,' + str(np.mean(train_time[encoding])) + ',' + str(np.std(train_time[encoding])) + '\n')




if __name__ == '__main__':
    main()