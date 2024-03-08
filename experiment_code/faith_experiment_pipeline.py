import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from faith import FAITH
# from loguru import logger
# logger.enable('flexfringe')

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

def main():
    # print('Reading data...')
    datasets = ['ctu', 'elastic', 'ugr']
    model_types = ['markov_chain', 'sm']
    input_folder = 'data/'
    encodings = ['ENCODE', 'PERCENTILE', 'FREQUENCY']

    ### For robustness experiment ###
     # input_folder = 'robustness_experiment_data'
    # robustness_types = ['padding', 'random_replacement', 'window_replacement', 'frequency_replacement']

    progress = tqdm(total= len(datasets) * len(model_types) * len(encodings) * 10, desc='Progress')
    for dataset in datasets:
        for model in model_types:
            auc_scores = dict()
            train_time = dict()
            
            for encoding in encodings:
                for i in range(10):
                    faith = FAITH(model, input_folder, 'ini/')

                    if encoding not in auc_scores:
                        auc_scores[encoding] = []
                        train_time[encoding] = []


                    training_file = input_folder + dataset + '_train_data_' + encoding + '.csv'
                    test_file = input_folder + dataset + '_test_data_' + encoding + '.csv'
                    
                    start = time.time()
                    faith.learn_model(training_file)
                    end = time.time()
                    train_time[encoding].append(end - start)
                    faith.compute_training_statistics(training_file)
                    test_predictions = faith.investigate_data(test_file)

            
                    # print('Computing AUC...')
                    # Compute AUC
                    y_true = test_predictions['type'].tolist()
                    y_true = [0 if x == ' benign' else 1 for x in y_true]
                    y_score = test_predictions['anomaly_score'].tolist()
                    # fpr, tpr, _ = roc_curve(y_true,  y_score)
                    auc = roc_auc_score(y_true, y_score)
                    auc_scores[encoding].append(auc)
                    progress.update(1)

                    # print('Plotting ROC curve')
                    # #create ROC curve
                    # _, ax = plt.subplots()
                    # add_identity(ax, color='gray', ls='--')
                    # plt.plot(fpr,tpr,label="AUC="+str(auc))
                    # plt.ylabel('True Positive Rate')
                    # plt.xlabel('False Positive Rate')
                    # plt.xlim(0, 1)
                    # plt.ylim(0, 1)
                    # plt.legend(loc=4)
                    # plt.savefig(input_folder + dataset + '_' + encoding + '_' + model + '_rolling_score_roc.png')
                    # plt.clf()
                    # plt.cla()

            # print('Writing AUC scores to file...')
            with open(input_folder + dataset + '_' + model + '_auc_scores_10_runs_rolling.csv', 'a') as f:
                f.write('encoding,AVG,STD\n')
                for encoding in auc_scores:
                    f.write(encoding + ',' + str(np.mean(auc_scores[encoding])) + ',' + str(np.std(auc_scores[encoding])) + '\n')
            
            # print('Writing training time to file...')
            with open(input_folder + dataset + '_' + model + '_train_time_10_runs_rolling.csv', 'a') as f:
                f.write('training_time,AVG,STD\n')
                for encoding in train_time:
                    f.write(encoding + ',' + str(np.mean(train_time[encoding])) + ',' + str(np.std(train_time[encoding])) + '\n')


if __name__ == '__main__':
    main()