from collections import Counter
import pandas as pd

def compute_rc_from_anomaly_scores(anomaly_scores: list, state_sequences: list) -> (list, list):
    """
    Compute the root causes of the traces based on their corresponding sequence anomaly scores.
    As described in the paper, the root cause of a anomaly for a given trace is the state that
    produced the highest anomaly score in the sequence of state transitions.

    :param anomaly_scores: The anomaly scores of the traces.
    :param state_sequences: The state sequences of the traces.
    """
    root_causes = []
    rc_indices = []
    for i in range(len(anomaly_scores)):
        rc_index = anomaly_scores[i].index(max(anomaly_scores[i]))
        rc = state_sequences[i][rc_index]
        root_causes.append(rc)
        rc_indices.append(rc_index)
    return root_causes, rc_indices

def collect_traces_by_rc(prediction_results: pd.DataFrame, rc: str) -> pd.DataFrame:
    """
    Collect traces with the same root cause symnbol. 

    :param prediction_results: The prediction results generated by FlexFringe.
    """
    return prediction_results[prediction_results['root_cause_symbol'] == rc]

def find_flows_for_anomalies(grouped_prediction_results: pd.DataFrame, netflow_data:pd.DataFrame) -> pd.DataFrame:
    """
    Find the (starting) NetFlow records for anomalies using the row numbers outputted in the 
    prediction results. FAITH learns a model from NetFlow data, grouped by their corresponding
    connection. This enables to only report the starting NetFlow of the connection producing 
    malicious behavior, and we thus do not use the indices returned by the 
    "compute_rc_from_anomaly_scores" function. In he case that a model is learned from NetFlow
    data sorted based on their timestamps, then we can use the indices to find the exact NetFlow
    records that produced the most anomalous behavior. 

    :param grouped_prediction_results: The prediction results grouped by anomaly.
    :param netflow_data: The NetFlow data used for prediction.
    """
    start_row_nrs = grouped_prediction_results['row nr'].tolist()
    return netflow_data[netflow_data.index.isin(start_row_nrs)]