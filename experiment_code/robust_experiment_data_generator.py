from collections import Counter
import pandas as pd 
import numpy as np
import os
from tqdm import tqdm
from copy import deepcopy


def find_nearest_feature_value(value, feature_values):
    return feature_values.index(min(feature_values, key=lambda x:abs(x-value)))

def randomly_select_categorical_value(x, feature_values):
    if x in feature_values:
        return x
    else:
        return np.random.choice(feature_values)

def find_suitable_benign_connections(benign_connections_sizes, window_size):
    return [connection for connection in benign_connections_sizes.keys() if benign_connections_sizes[connection] >= window_size]

def find_nearest_flow(flow, flow_list):
    distances = [np.linalg.norm(flow - x) for x in flow_list]
    return distances.index(min(distances))


def pad_data(train_data, test_data):
    benign_bytes = train_data['num_bytes'].tolist()
    benign_packets = train_data['num_packets'].tolist()
    benign_protos = train_data['proto'].tolist()
    benign_durs = train_data['dur'].tolist()
    bening_sport = train_data['sport'].tolist()
    benign_dport = train_data['dport'].tolist()
    benign_flags = train_data['flags'].tolist()
    benign_fwd_stat = train_data['fwd_stat'].tolist()
    benign_tos = train_data['ToS'].tolist()
    # bening_src_bytes = train_data['SrcBytes'].tolist()
    # benign_dTos = train_data['dTos'].tolist()
    # benign_bytes_encoding = train_data['bytes_encoding_conn'].tolist()
    # benign_packets_encoding = train_data['packets_encoding_conn'].tolist()

    test_benign_data = test_data[test_data['label'] == 'benign']
    malicious_test_data = test_data[test_data['label'] == 'malicious']
    malicious_bytes = deepcopy(malicious_test_data['num_bytes'].tolist())
    malicious_packets = deepcopy(malicious_test_data['num_packets'].tolist())
    malicious_protos = deepcopy(malicious_test_data['proto'].tolist())
    malicious_durs = deepcopy(malicious_test_data['dur'].tolist())
    malicious_sport = deepcopy(malicious_test_data['sport'].tolist())
    malicious_dport = deepcopy(malicious_test_data['dport'].tolist())
    malicious_flags = deepcopy(malicious_test_data['flags'].tolist())
    malicious_fwd_stat = deepcopy(malicious_test_data['fwd_stat'].tolist())
    malicious_tos = deepcopy(malicious_test_data['ToS'].tolist())

    # malicious_src_bytes = deepcopy(malicious_test_data['SrcBytes'].tolist())
    # malicious_dTos = deepcopy(malicious_test_data['dTos'].tolist())

    nearest_benign_bytes = [find_nearest_feature_value(x, benign_bytes) for x in malicious_bytes]
    nearest_benign_packets = [find_nearest_feature_value(x, benign_packets) for x in malicious_packets]
    nearest_benign_durs = [find_nearest_feature_value(x, benign_durs) for x in malicious_durs]
    nearest_benign_sport = [find_nearest_feature_value(x, bening_sport) for x in malicious_sport]
    nearest_benign_dport = [find_nearest_feature_value(x, benign_dport) for x in malicious_dport]
    # nearest_benign_src_bytes = [find_nearest_feature_value(x, bening_src_bytes) for x in malicious_src_bytes]
    # nearest_benign_dTos = [find_nearest_feature_value(x, benign_dTos) for x in malicious_dTos]

    padded_bytes = [benign_bytes[x] for x in nearest_benign_bytes]
    padded_packets = [benign_packets[x] for x in nearest_benign_packets]
    padded_durs = [benign_durs[x] for x in nearest_benign_durs]
    padded_sport = [bening_sport[x] for x in nearest_benign_sport]
    padded_dport = [benign_dport[x] for x in nearest_benign_dport]
    # padded_src_bytes = [bening_src_bytes[x] for x in nearest_benign_src_bytes]
    # padded_dTos = [benign_dTos[x] for x in nearest_benign_dTos]


    # updated_bytes_encodings = [benign_bytes_encoding[x] for x in nearest_benign_bytes]
    # updated_packets_encodings = [benign_packets_encoding[x] for x in nearest_benign_packets]
    # updated_durs_encodings = [benign_packets_encoding[x] for x in nearest_benign_durs]

    padded_protos = [randomly_select_categorical_value(x, benign_protos) for x in malicious_protos]
    padded_flags = [randomly_select_categorical_value(x, benign_flags) for x in malicious_flags]
    padded_fwd_stat = [randomly_select_categorical_value(x, benign_fwd_stat) for x in malicious_fwd_stat]
    padded_tos = [randomly_select_categorical_value(x, benign_tos) for x in malicious_tos]

    malicious_test_data['num_bytes'] = padded_bytes
    malicious_test_data['num_packets'] = padded_packets
    malicious_test_data['dur'] = padded_durs
    malicious_test_data['proto'] = padded_protos
    malicious_test_data['sport'] = padded_sport
    malicious_test_data['dport'] = padded_dport
    malicious_test_data['flags'] = padded_flags
    malicious_test_data['fwd_stat'] = padded_fwd_stat
    malicious_test_data['ToS'] = padded_tos
    # malicious_test_data['SrcBytes'] = padded_src_bytes
    # malicious_test_data['dTos'] = padded_dTos
    # malicious_test_data['bytes_encoding_conn'] = updated_bytes_encodings
    # malicious_test_data['packets_encoding_conn'] = updated_packets_encodings
    # malicious_test_data['duration_encoding_conn'] = updated_durs_encodings

    return pd.concat([test_benign_data, malicious_test_data])


def random_replace(train_data, test_data):
    benign_data = train_data.values.tolist()
    benign_bytes = train_data['num_bytes'].tolist()
    benign_packets = train_data['num_packets'].tolist()
    benign_durs = train_data['dur'].tolist()
    benign_protos = train_data['proto'].tolist()
    benign_sport = train_data['sport'].tolist()
    benign_dport = train_data['dport'].tolist()
    benign_flags = train_data['flags'].tolist()
    benign_fwd_stat = train_data['fwd_stat'].tolist()
    benign_tos = train_data['ToS'].tolist()
    # benign_src_bytes = train_data['SrcBytes'].tolist()
    # benign_dTos = train_data['dTos'].tolist()

    # benign_bytes_encoding = train_data['bytes_encoding_conn'].tolist()
    # benign_packets_encoding = train_data['packets_encoding_conn'].tolist()
    # benign_duration_encoding = train_data['duration_encoding_conn'].tolist()

    test_benign_data = test_data[test_data['label'] == 'benign']
    malicious_test_data = test_data[test_data['label'] == 'malicious']
    malicious_bytes = deepcopy(malicious_test_data['num_bytes'].tolist())
    malicious_packets = deepcopy(malicious_test_data['num_packets'].tolist())
    malicious_protos = deepcopy(malicious_test_data['proto'].tolist())
    malicious_dur = deepcopy(malicious_test_data['dur'].tolist())
    malicious_sport = deepcopy(malicious_test_data['sport'].tolist())
    malicious_dport = deepcopy(malicious_test_data['dport'].tolist())
    malicious_flags = deepcopy(malicious_test_data['flags'].tolist())
    malicious_fwd_stat = deepcopy(malicious_test_data['fwd_stat'].tolist())
    malicious_tos = deepcopy(malicious_test_data['ToS'].tolist())
    # malicious_src_bytes = deepcopy(malicious_test_data['SrcBytes'].tolist())
    # malicious_dTos = deepcopy(malicious_test_data['dTos'].tolist())

    # malicious_bytes_encoding = deepcopy(malicious_test_data['bytes_encoding_conn'].tolist())
    # malicious_packets_encoding = deepcopy(malicious_test_data['packets_encoding_conn'].tolist())
    # malicious_duration_encoding = deepcopy(malicious_test_data['duration_encoding_conn'].tolist())
    

    for i in range(len(malicious_bytes)):
        # 15% chance of replacement
        if np.random.random() < 0.15:
            random_index = np.random.randint(0, len(benign_data))
            malicious_bytes[i] = benign_bytes[random_index]
            malicious_packets[i] = benign_packets[random_index]
            malicious_protos[i] = benign_protos[random_index]
            malicious_dur[i] = benign_durs[random_index]
            malicious_sport[i] = benign_sport[random_index]
            malicious_dport[i] = benign_dport[random_index]
            malicious_flags[i] = benign_flags[random_index]
            malicious_fwd_stat[i] = benign_fwd_stat[random_index]
            malicious_tos[i] = benign_tos[random_index]
            # malicious_src_bytes[i] = benign_src_bytes[random_index]
            # malicious_dTos[i] = benign_dTos[random_index]
            # malicious_bytes_encoding[i] = benign_bytes_encoding[random_index]
            # malicious_packets_encoding[i] = benign_packets_encoding[random_index]
            # malicious_duration_encoding[i] = benign_duration_encoding[random_index]

    

    malicious_test_data['num_bytes'] = malicious_bytes
    malicious_test_data['num_packets'] = malicious_packets
    malicious_test_data['dur'] = malicious_dur
    malicious_test_data['proto'] = malicious_protos
    malicious_test_data['sport'] = malicious_sport
    malicious_test_data['dport'] = malicious_dport
    malicious_test_data['flags'] = malicious_flags
    malicious_test_data['fwd_stat'] = malicious_fwd_stat
    malicious_test_data['ToS'] = malicious_tos
    # malicious_test_data['SrcBytes'] = malicious_src_bytes
    # malicious_test_data['dTos'] = malicious_dTos
    # malicious_test_data['bytes_encoding_conn'] = malicious_bytes_encoding
    # malicious_test_data['packets_encoding_conn'] = malicious_packets_encoding
    # malicious_test_data['duration_encoding_conn'] = malicious_duration_encoding

    return pd.concat([test_benign_data, pd.DataFrame(malicious_test_data, columns=test_data.columns)])

def window_replace(train_data, test_data, window_size):
    # grouped connection data
    grouped_connection_train_data = train_data.groupby(['src_ip', 'dst_ip'])
    test_benign_data = test_data[test_data['label'] == 'benign']
    grouped_malicious_test_data = test_data[test_data['label'] == 'malicious'].groupby(['src_ip', 'dst_ip'])

    malcon_size_mapping = dict()
    bencon_size_mapping = dict()
    malcon_bencon_mapping = dict()

    malcon_size_mapping = {malicious_group: grouped_malicious_test_data.get_group(malicious_group).shape[0] for malicious_group in grouped_malicious_test_data.groups.keys()}
    bencon_size_mapping = {benign_group: grouped_connection_train_data.get_group(benign_group).shape[0] for benign_group in grouped_connection_train_data.groups.keys()}

    for malicious_group in tqdm(malcon_size_mapping.keys(), desc='finding suitable connections'):
        malcon_bencon_mapping[malicious_group] = find_suitable_benign_connections(bencon_size_mapping, window_size)

    selected_bencon = dict()
    for malicious_group in tqdm(malcon_size_mapping.keys(), desc='selecting connections'):
        if malicious_group not in selected_bencon:
            selected_bencon[malicious_group] = []
        
        suitable_connections = malcon_bencon_mapping[malicious_group]
        mal_size = malcon_size_mapping[malicious_group]
        selected_size = 0
        while selected_size < mal_size:
            connection = suitable_connections[np.random.randint(0, len(suitable_connections))]
            connection_size = bencon_size_mapping[connection]
            selected_size += connection_size
            selected_bencon[malicious_group].append(connection)
    
    sampled_benign_data = dict()
    for malicious_group in tqdm(malcon_size_mapping.keys(), desc='sampling data'):
        if malicious_group not in sampled_benign_data:
            sampled_benign_data[malicious_group] = []

        selected_connections = selected_bencon[malicious_group]
        for connection in selected_connections:
            sampled_benign_data[malicious_group].append(grouped_connection_train_data.get_group(connection))
        
        sampled_benign_data[malicious_group] = pd.concat(sampled_benign_data[malicious_group])
    
    replaced_malicious_test_data = []

    for malicious_group in tqdm(malcon_size_mapping.keys(), desc='replacing data'):
        sampled_conn_data = sampled_benign_data[malicious_group].head(n=malcon_size_mapping[malicious_group])
        group_malicious_data = grouped_malicious_test_data.get_group(malicious_group)
        replaced_malicious_group_data = pd.DataFrame(columns=test_data.columns)
        replaced_malicious_group_data['timestamp'] = deepcopy(group_malicious_data['timestamp'].tolist())
        replaced_malicious_group_data['src_ip'] = deepcopy(group_malicious_data['src_ip'].tolist())
        replaced_malicious_group_data['dst_ip'] = deepcopy(group_malicious_data['dst_ip'].tolist())
        replaced_malicious_group_data['label'] = deepcopy(group_malicious_data['label'].tolist())
        replaced_malicious_group_data['num_bytes'] = deepcopy(sampled_conn_data['num_bytes'].tolist())
        replaced_malicious_group_data['num_packets'] = deepcopy(sampled_conn_data['num_packets'].tolist())
        replaced_malicious_group_data['dur'] = deepcopy(sampled_conn_data['dur'].tolist())
        replaced_malicious_group_data['proto'] = deepcopy(sampled_conn_data['proto'].tolist())
        replaced_malicious_group_data['sport'] = deepcopy(sampled_conn_data['sport'].tolist())
        replaced_malicious_group_data['dport'] = deepcopy(sampled_conn_data['dport'].tolist())
        replaced_malicious_group_data['flags'] = deepcopy(sampled_conn_data['flags'].tolist())
        replaced_malicious_group_data['fwd_stat'] = deepcopy(sampled_conn_data['fwd_stat'].tolist())
        replaced_malicious_group_data['ToS'] = deepcopy(sampled_conn_data['ToS'].tolist())
        # replaced_malicious_group_data['SrcBytes'] = deepcopy(sampled_conn_data['SrcBytes'].tolist())
        # replaced_malicious_group_data['dTos'] = deepcopy(sampled_conn_data['dTos'].tolist())
        # replaced_malicious_group_data[':bytes_encoding_conn'] = deepcopy(sampled_conn_data[':bytes_encoding_conn'].tolist())
        # replaced_malicious_group_data[':packets_encoding_conn'] = deepcopy(sampled_conn_data[':packets_encoding_conn'].tolist())
        # replaced_malicious_group_data[':duration_encoding_conn'] = deepcopy(sampled_conn_data[':duration_encoding_conn'].tolist())
        replaced_malicious_test_data.append(replaced_malicious_group_data)
    
    replaced_malicious_test_data = pd.concat(replaced_malicious_test_data)

    return pd.concat([test_benign_data, replaced_malicious_test_data])


def frequency_replacement(train_data, test_data, features, frequency_threshold):
    benign_train_features_data = train_data[features].values.tolist()
    # benign_train_flags_data = train_data['flags'].tolist()
    # benign_train_proto_data = train_data['symb:proto'].tolist()
    # benign_flags_counter = Counter(benign_train_flags_data)
    # bening_proto_counter = Counter(benign_train_proto_data)
    # most_frequent_flag = benign_flags_counter.most_common(1)[0][0]
    # most_frequent_proto = bening_proto_counter.most_common(1)[0][0]
    test_malicious_data = test_data[test_data['type:label'] == 'malicious']
    test_benign_data = test_data[test_data['type:label'] == 'benign']
    benign_train_features_data_str = [','.join([str(x) for x in row]) for row in benign_train_features_data]
    benign_flows_freq = Counter(benign_train_features_data_str)
    frequent_unique_flows = {flow: freq for flow, freq in benign_flows_freq.items() if freq >= frequency_threshold}
    frequent_benign_flows_list = [flow.split(',') for flow in frequent_unique_flows.keys()]
    frequent_benign_flows_df = pd.DataFrame(frequent_benign_flows_list, columns=features)
    frequent_benign_flows_df = frequent_benign_flows_df.astype({
                                                                'symb:proto': 'int64', 
                                                                'symb:bytes_encoding_conn': 'int64', 
                                                                'symb:packets_encoding_conn': 'int64', 
                                                                'symb:duration_encoding_conn': 'int64',
                                                                'num_bytes': 'int64',
                                                                'num_packets': 'int64',
                                                                'dur': 'float64',
                                                                # 'SrcBytes': 'int64',
                                                                # 'dTos': 'float64',
                                                                # 'sport' : 'int64',
                                                                # 'dport': 'int64',
                                                                # 'ToS': 'int64',
                                                                # 'fwd_stat': 'int64'
                                                            })
    
    frequent_bening_flows_df = frequent_benign_flows_df.drop(['id:src_ip', 'id:dst_ip'], axis=1)
    frequent_benign_flows_np = frequent_bening_flows_df.values

    test_malicious_features_data_df = test_malicious_data[features]
    test_malicious_features_data_df = test_malicious_features_data_df.drop(['id:src_ip', 'id:dst_ip'], axis=1)
    test_malicious_src_and_dst = test_malicious_data[['id:src_ip', 'id:dst_ip']].values.tolist()
    test_malicious_features_data_np = test_malicious_features_data_df.values

    replaced_malicious_flows = [] 
    # replaced_malicious_flows_flags = []
    # replaced_malicious_flows_proto = []
    for i in range(len(test_malicious_features_data_np)):
        flow  = test_malicious_features_data_np[i]
        nearest_frequent_benign_flow_index = find_nearest_flow(flow, frequent_benign_flows_np)
        nearest_frequent_benign_flow = frequent_benign_flows_list[nearest_frequent_benign_flow_index]
        nearest_frequent_benign_flow[0] = test_malicious_src_and_dst[i][0]
        nearest_frequent_benign_flow[1] = test_malicious_src_and_dst[i][1]
        replaced_malicious_flows.append(nearest_frequent_benign_flow)
        # replaced_malicious_flows_flags.append(most_frequent_flag)
        # replaced_malicious_flows_proto.append(most_frequent_proto)

    replaced_malicious_flows_df = pd.DataFrame(replaced_malicious_flows, columns=features)
    replaced_malicious_flows_df = replaced_malicious_flows_df.astype({
                                                                    'symb:proto': 'int64',
                                                                    'symb:bytes_encoding_conn': 'int64', 
                                                                    'symb:packets_encoding_conn': 'int64', 
                                                                    'symb:duration_encoding_conn': 'int64',
                                                                    'num_bytes': 'int64',
                                                                    'num_packets': 'int64',
                                                                    'dur': 'float64',
                                                                    # 'SrcBytes': 'int64',
                                                                    # 'dTos': 'float64',
                                                                    # 'sport' : 'int64',
                                                                    # 'dport': 'int64',
                                                                    # 'ToS': 'int64',
                                                                    # 'fwd_stat': 'int64'
                                                                })
    replaced_malicious_flows_df['timestamp'] = test_malicious_data['timestamp'].tolist()
    # replaced_malicious_flows_df['flags'] = replaced_malicious_flows_flags
    # replaced_malicious_flows_df['symb:proto'] = replaced_malicious_flows_proto
    replaced_malicious_flows_df['type:label'] = ['malicious'] * len(replaced_malicious_flows)
    test_benign_data = test_benign_data[ ['timestamp'] + features + ['type:label']]
    return pd.concat([test_benign_data, replaced_malicious_flows_df])


    

def main():
    datasets = ['elastic', 'ugr', 'ctu']
    input_folder = 'data/'
    robustness_types = ['padding', 'random_replacement', 'window_replacement', 'frequency_replacement']
    # make outputfolder 
    output_folder = 'robustness_experiment_data'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    progress = tqdm(total= len(datasets) * len(robustness_types), desc='Progress')
    FAITH_features = ['id:src_ip', 'id:dst_ip', 'symb:proto', 'num_bytes', 'num_packets', 'dur', 'symb:bytes_encoding_conn', 'symb:packets_encoding_conn', 'symb:duration_encoding_conn']
    ZOPPI_features = ['src_ip', 'dst_ip', 'dur', 'SrcBytes', 'dTos']
    RELOAD_features = ['src_ip', 'dst_ip', 'SrcBytes', 'sport']
    GEE_features = ['dur', 'src_ip', 'dst_ip', 'sport', 'dport', 'num_bytes', 'num_packets', 'fwd_stat', 'ToS']

    for dataset in datasets:
        train_data = pd.read_csv(input_folder + '/' + dataset + '_train_data_ENCODE.csv', low_memory=False)
        # train_data['label'] = train_data['label'].apply(lambda x: 'benign' if x == 'background' else 'malicious')
        column_order = train_data.columns.tolist()
        test_data = pd.read_csv(input_folder + '/' + dataset + '_test_data_ENCODE.csv', low_memory=False)
        # test_data['label'] = test_data['label'].apply(lambda x: 'benign' if x == 'background' else 'malicious')

        for robustness_type in robustness_types:
            if robustness_type == 'padding':
                new_test_data = pad_data(train_data, test_data)
                new_test_data = new_test_data[column_order]
                new_test_data.to_csv(output_folder + '/' + dataset.upper() + '_test_data_FAITH_features_robust_padding.csv', sep=',', index=False)
            elif robustness_type == 'random_replacement':
                new_test_data = random_replace(train_data, test_data)
                new_test_data = new_test_data[column_order]
                new_test_data.to_csv(output_folder + '/' + dataset.upper() + '_test_data_FAITH_features_robust_random_replacement.csv', sep=',', index=False)
            elif robustness_type == 'window_replacement':
                new_test_data = window_replace(train_data, test_data, 10)
                new_test_data = new_test_data[column_order]
                new_test_data.to_csv(output_folder + '/' + dataset.upper() + '_test_data_FAITH_features_robust_window_replacement.csv', sep=',', index=False)
            elif robustness_type == 'frequency_replacement':
                new_test_data = frequency_replacement(train_data, test_data, FAITH_features, 100)
                new_test_data = new_test_data[column_order]
                new_test_data.to_csv(output_folder + '/' + dataset.upper() + '_test_data_FAITH_features_robust_frequency_replacement.csv', sep=',', index=False)

            progress.update(1)


if __name__ == '__main__':
    main()

