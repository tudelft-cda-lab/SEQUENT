from collections import Counter
import math
import pandas as pd

def load_result_file(filename: str) -> pd.DataFrame:
    """
    Load the result file produced by FlexFringe after running the predictions on the learned model.
    The results file is returned as a Pandas DataFrame.

    :param filename: The path to the result file.
    """
    data = pd.read_csv(filename, delimiter='; ')
    return data

def state_sequence_to_list(state_sequence: str) -> list:
    """
    Convert a state sequence (stored as a string) to a list object.

    :param state_sequence: The state sequence that should be converted to a list.
    """
    return state_sequence.replace('[', '').replace(']', '').strip().split(',')

def compute_state_frequencies(state_sequences: list) -> Counter:
    """
    Compute the state visit frquencies from a list of state sequences.
    """
    states = []
    for state_sequence in state_sequences:
        states += state_sequence

    return Counter(states)

def compute_rolling_anomaly_score(test_state_sequences: list, state_frequency_train: dict) -> (list, list):
    """
    Compute the rolling anomaly score as described in the paper "Frequency-based Network Anomaly Detection Using State Machines".
    The anomaly score adapts itself at test-time based on the state visit frequencies observed in the test traces. 

    :param test_state_sequences: The state sequences extracted after running predictions on test traces.
    :param state_frequency_train: The state visit frequencies computed from the training data.
    """
    anomaly_scores = []
    sequence_anomaly_scores = []
    test_state_counts = Counter()
    total_count_from_train = sum(state_frequency_train.values()) # used for nomalization.

    for state_sequence in test_state_sequences:
        sequence_score = 0.0
        seq_anom_score = []
        for state in state_sequence:
            state_score = math.log(test_state_counts[state] + 1)
            state_score -= math.log(state_frequency_train[state] + 1)
            state_score += math.log(total_count_from_train + 1)
            state_score -= math.log(len(test_state_counts.keys()) + 1)
            test_state_counts[state] += 1
            sequence_score += state_score
            seq_anom_score.append(state_score)
        
        sequence_anomaly_scores.append(seq_anom_score)
        anomaly_scores.append(sequence_score)

    return anomaly_scores, sequence_anomaly_scores