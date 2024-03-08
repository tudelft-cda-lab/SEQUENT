from faith.ff_result_processor import *
from faith.model_operations import *
from faith.root_cause_operations import *
import pandas as pd 

class FAITH:
    def __init__(self, model:str, output_folder: str, ini_folder:str):
        """
        Contructs an object of the FAITH class.

        :param model: The type of model that should be learned using FAITH. Currently, we only support Markov Chains and State Machines 
        :param output_path: The path where the output files of FAITH should be stored.
        :param ini_folder: The path to the folder containing the INI files for training and running predictions with FlexFringe.
        """
        if model not in ['markov_chain', 'sm']:
            raise ValueError('The model type provided is not supported. Please choose one of the following: markov_chain, sm')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        if not os.path.exists(ini_folder) or ini_folder == None:
            raise ValueError('The provided INI folder does not exist. Please provide a valid path to the INI folder.')
        
        if not os.path.exists(ini_folder + model + '.ini'):
            raise ValueError('No INI file for training could be found for the provided model type. Please make sure that a INI file exists (in the provided INI folder) with the name: ' + model + '.ini')

        if not os.path.exists(ini_folder + model + '_predict.ini'):
            raise ValueError('No INI file for prediction could be found for the provided model type. Please make sure that a INI file exists (in the provided INI folder) with the name: ' + model + '_predict.ini')

        self.model = model
        self.ini_folder = ini_folder
        self.ini_for_learning = ini_folder + model + '.ini'
        self.ini_for_prediction = ini_folder + model + '_predict.ini'
        self.output_folder = output_folder

    def learn_model(self, data_path: str):
        """
        Learns a model (in an unupervised manner) on the data file provided in the "data_path" parameter.
        This function makes a call to the fit from the python wrapper of FlexFringe, which subsequently 
        initiates the learning process of FlexFringe. This function also extracts some extra information
        that will be used later for the investigation of (new) data.

        :param data_path: The path to the data file that should be used for learning a model.
        """
        ff_output_file = self.output_folder + self.model + '_train.ff'
        output_format = 'json'
        self.model_file = ff_output_file + '.final.' + output_format # store the location of the model file for later use (predictions).
        train(data_path , ff_output_file, output_format, self.ini_for_learning)


    def compute_training_statistics(self, data_path: str):
        """
        Compute the state visits frequencies from the training data. This information is used later on for the
        investigation of (new) data. Specifically, the statistics are used for computing rolling anomaly scores
        of traces. 

        :param data_path: The path to the training data that should be used for computing the state visit frequencies.
        """
        training_predictions = predict(data_path , self.model_file, self.ini_for_prediction)
        self.train_state_counts = compute_state_frequencies(training_predictions['state sequence'].tolist()) # store the state frequencies for computing anomaly scores later on.

    def investigate_data(self, data_path: str) -> pd.DataFrame:
        """
        Do an investigation of the data file provided in the "data_path" parameter and compute the anomaly 
        scores (including root causes) for the traces extracted from the data file. This function makes a call
        to the predict function from the python wrapper of FlexFringe, which subsequently triggers the simulation
        of the traces on the learned model. The anomaly scores are computed using the compute_rolling_anomaly_score
        function and the root causes are computed using the compute_rc_from_anomaly_scores function.

        :param data_path: The path to the data file that should be used for the investigation.
        """
        test_predictions = predict(data_path, self.model_file, self.ini_for_prediction)
        test_state_sequences = test_predictions['state sequence'].tolist()
        anomaly_scores, sequence_anomaly_scores = compute_rolling_anomaly_score(test_state_sequences, self.train_state_counts)
        test_predictions['sequence_anomaly_score'] = sequence_anomaly_scores
        test_predictions['anomaly_score'] = anomaly_scores
        test_predictions['root_cause_symbol'] = [compute_rc_from_anomaly_scores([x], [y])[0][0] for x, y in zip(sequence_anomaly_scores, test_state_sequences)]
        selected_columns_for_output = [
            'row nr',
            'last row nr',
            'abbadingo trace',
            'state sequence',
            'sequence_anomaly_score',
            'anomaly_score',
            'root_cause_symbol',
            'sum scores',
            'type' # this field should only be used for testing purposes. In practice, this field is not available. 
        ]

        # first convert root cause symbols to strings.
        test_predictions['root_cause_symbol'] = test_predictions['root_cause_symbol'].astype(str)
        return test_predictions[selected_columns_for_output]

    
    def group_predictions(prediction_results: pd.DataFrame, netflow_records: pd.DataFrame) -> dict():
        """
        Group the prediction results by their root cause symbols. We first rank the the groups by their sizes (largest first)
        and then we sort the traces of each group by their anomaly scores (largest first). We then link the traces of each
        group to the corresponding NetFlow records. We store all these results in a list and return it.

        :param prediction_results: The prediction results generated by FAITH.
        :param netflow_records: The NetFlow records used for prediction.
        """
        group_size_mapping = dict()
        group_data_mapping = dict()
        grouped_prediction_results = prediction_results.groupby('root_cause_symbol')
        for g in grouped_prediction_results.groups.keys():
            group_results = grouped_prediction_results.get_group(g)
            group_size = len(group_results.index)
            group_size_mapping[g] = group_size
            group_results = group_results.sort_values(by=['anomaly_score'], ascending=False) # sort the traces by their anomaly scores (largest first).
            linked_netflow_records = find_flows_for_anomalies(group_results, netflow_records) # link traces to NetFlow records.
            group_data_mapping[g] = {'prediction_results': group_results, 'linked_netflow': linked_netflow_records}
        
        # sort the groups by their size (largest first).
        sorted_anomaly_groups = []
        sorted_group_size_mapping = sorted(group_size_mapping.items(), key=lambda x: x[1], reverse=True)
        for g in sorted_group_size_mapping:
            sorted_anomaly_groups.append((g[0],group_data_mapping[g[0]]))
        
        return sorted_anomaly_groups




            


