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
    print('Reading data...')
    datasets = ['ctu', 'elastic', 'ugr']
    encodings = ['ENCODE', 'PERCENTILE', 'FREQUENCY']
    input_folder = 'data/'
    model_type = 'IF'

    ### For robustness experiment ###
    # input_folder = 'robustness_experiment_data'
    # robustness_types = ['frequency_replacement']#'padding', 'random_replacement', 'window_replacement']#, 'frequency_replacement']

    progress_bar = tqdm(total= len(datasets) * len(encodings) * 10, desc='Progress')

    for dataset in datasets:
        auc_scores = dict()
        train_time = dict()

        for encoding in encodings:
            if encoding not in auc_scores:
                auc_scores[encoding] = []
                train_time[encoding] = []

            for i in range(10):
                ### Dataset containing only FAITH features #####
                # training_file = input_folder + dataset + '_train_data_' + encoding + '.csv'
                # test_file = input_folder + dataset + '_test_data_' + encoding + '.csv'
                
                ####### Dataset with ZOPPI features ########
                training_file = input_folder + dataset + '_train_data_zoppi_features.csv'
                test_file = input_folder + dataset + '_test_data_zoppi_features.csv'

                #### FAITH features #####
                # raw_feature_columns = ['dur', 'symb:proto', 'num_bytes', 'num_packets'] 

                #### ZOPPI features #####
                raw_feature_columns = ['dur', 'SrcBytes', 'dTos'] # original zoppi features for CTU (IF)

                #### Features selection applied for FastABOD (Zoppi) using RELOAD #####
                # raw_feature_columns = ['SrcBytes','timestamp','sport'] 

                #### FAITH features, where the bytes, packets and duration are discretized using ENCODE #####
                # encoded_feature_columns = ['symb:proto', 'symb:bytes_encoding_conn', 'symb:packets_encoding_conn', 'symb:duration_encoding_conn']

                train_data = pd.read_csv(training_file)
                test_data = pd.read_csv(test_file)

                y_true = test_data['label'].tolist()
                y_true = [0 if x == 'benign' else 1 for x in y_true]

                train_data = train_data[raw_feature_columns]
                test_data = test_data[raw_feature_columns]

                ### Use for discretized FAITH features ###
                # train_data = train_data[encoded_feature_columns]
                # test_data = test_data[encoded_feature_columns]

                # if encoding == 'FREQUENCY':
                #     for feature in encoded_feature_columns:
                #         if feature == 'symb:proto':
                #             continue
                #         else:
                #             train_data[feature] = tranform_symbols_for_frequency_encoding(train_data[feature].tolist())
                #             test_data[feature] = tranform_symbols_for_frequency_encoding(test_data[feature].tolist())

                # Use for ABOD
                if model_type == 'FastABOD':
                    train_data = train_data.astype('float64')
                    test_data = test_data.astype('float64')

                    for feature in raw_feature_columns:
                        train_data[feature] = add_normal_dist_noise_to_feat_data(train_data[feature].to_numpy())
                        test_data[feature] = add_normal_dist_noise_to_feat_data(test_data[feature].to_numpy())
                    
                    
                    model = ABOD(n_neighbors=5, contamination=0.01, method='fast')

                elif model_type == 'IF':
                    model = IForest(n_estimators=5, max_samples=50 , contamination=0.01, n_jobs=-1, verbose=0)
                
                elif model_type == 'HBOS':
                     model = HBOS(n_bins=10, alpha=0.1)

                start = time.time()
                model.fit(train_data)
                end = time.time()

                train_time[encoding].append(end - start)

                # print('Computing AUC...')
                # Compute AUC
                y_score = model.decision_function(test_data.values).tolist()
                auc = roc_auc_score(y_true, y_score)
                auc_scores[encoding].append(auc)
                progress_bar.update(1)

                # print('Plotting ROC curve')
                # create ROC curve
                # fpr, tpr, _ = roc_curve(y_true,  y_score)
                # _, ax = plt.subplots()
                # add_identity(ax, color='gray', ls='--')
                # plt.plot(fpr,tpr,label="AUC="+str(auc))
                # plt.ylabel('True Positive Rate')
                # plt.xlabel('False Positive Rate')
                # plt.xlim(0, 1)
                # plt.ylim(0, 1)
                # plt.legend(loc=4)
                # plt.savefig(input_folder + '/' + dataset + '_' + encoding + '_FastABOD_roc.png')
                # plt.clf()
                # plt.cla()
        
        print('Writing average and standard deviation of AUC scores to file...')
        with open(input_folder + dataset + '_' + model_type + '_auc_scores_10_runs_zoppi_features.csv', 'a') as f:
            f.write('ENCODING,AVG,STD\n')
            for encoding in encodings:
                f.write(encoding + ',' + str(np.mean(auc_scores[encoding])) + ',' + str(np.std(auc_scores[encoding])) + '\n')

        print('Writing average and standard deviation of training time to file...')
        with open(input_folder + dataset + '_' + model_type + '_train_time_10_runs_zoppi_features.csv', 'a') as f:
            f.write('ENCODING,AVG,STD\n')
            for encoding in encodings:
                f.write(encoding + ',' + str(np.mean(train_time[encoding])) + ',' + str(np.std(train_time[encoding])) + '\n')




if __name__ == '__main__':
    main()