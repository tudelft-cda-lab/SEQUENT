# SEQUENT: State Frequency Estimation for Anomaly Detection
[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## About 
SEQUENT is a frequency-based method that employs state machines for the detection of network anomalies. SEQUENT learns a state machine exclusively from benign NetFlow data to model sequential network behavior. You can view the states of a learned state machine model as latent values that are useful for predicting future network behavior. Each state in the model represents a different kind of behavior (NetFlow sequence) observed in the benign data. SEQUENT extracts state visit frequency from the learned model and uses it to compute adaptive anomaly scores by comparing it to the expected frequencies observed at train time. Thus, whenever there is a kind of behavior that occurs more frequently at test time than at train time, SEQUENT raises an alarm. The key intuition behind SEQUENT is that such frequency increases are signals for anomalous activities such as network attacks. 

A well-known challenge faced by many security analysts in their daily work routine is the large volume of alerts raised by the anomaly/intrusion detection system, turning the inspection of alerts to a labor-intensive task. The sheer volume of alerts makes it difficult for an analyst to prioritize which anomalies they should inspect first. SEQUENT addresses this issue by computing root causes for each detected anomaly. The root causes offer the ability to group and rank anomalies; once anomalies are grouped by their root causes, an analyst can use the frequency of the root causes to rank groups of anomalies. The analysts can then use the ranking to prioritize which group anomalies they should inspect. Additionally, SEQUENT establishes links between the root causes and the corresponding network data. The objective is to provide analysts with concrete instances of anomalous network behavior, enabling a quick and in-depth analysis of the discovered anomalies. 

## Installation
SEQUENT can be installed by first cloning this repository and then running the following command in your terminal:
```
pip install .
```

Alternatively, you can install SEQUENT by running the following command in your terminal:
```
pip install git+https://github.com/tudelft-cda-lab/SEQUENT.git
```
Running `pip install` should already install the required dependencies of SEQUENT. However, if you would like to use the exact version of dependencies that we use, you can install them by running the following command in your terminal:
```
pip install -r requirements.txt
```

As SEQUENT employs FlexFringe for the learning of models, make sure that you have downloaded from [here](https://github.com/tudelft-cda-lab/FlexFringe) and specify where the binary of FlexFringe is located on your system by executing the following command in your terminal:
```
export FLEXFRINGE_PATH=/path/to/flexfringe/binary
```

Furthermore, to make the interaction with FlexFringe a bit easier, SEQUENT utilizes the FlexFringe Python wrapper. To install the wrapper, execute the following command in your terminal:
```
pip install git+https://github.com/ClintonCao/FlexFringe-python.git
```

## Usage
The code below shows how to train SEQUENT and how to use the trained model to detect anomalies. You can also have a look at the experiment code on the `experiment_code` branch to see how we use SEQUENT to detect anomalies on the AssureMOSS, CTU-13, and UGR-16 datasets. 

```python

from SEQUENT import SEQUENT

# Initialize SEQUENT to train a state machine model
SEQUENT = SEQUENT('sm', 'path/to/output/directory')
training_data = 'path/to/training/data'
SEQUENT.learn_model(training_data)
SEQUENT.compute_training_statistics(training_data)

# Run SEQUENT on test data
test_data = 'path/to/test/data'
prediction_results = SEQUENT.investigate_data(test_data)

"""
Group predictions by their root causes, rank the groups by their sizes, and link the traces to the corresponding NetFlow data.
The returned sorted_anomaly_groups is a list of tuples, where each tuple contains the root cause, the size of the group, and 
the traces of the group. The groups are sorted by their sizes in descending order.
"""
sorted_anomaly_groups = SEQUENT.group_predictions(prediction_results, test_data)

```

## Citation
If you use SEQUENT in your research, please cite the following paper:
```
@misc{cao2023SEQUENT,
      title={SEQUENT: Frequency-based Network Anomaly Detection Using State Machines}, 
      author={Clinton Cao, Agathe Blaise, Sicco Verwer, and Annibale Panichella},
      year={2023},
      eprint={TBD},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contact
If you have any questions, feel free to drop me (Clinton) an email. My email address is listed on my GitHub page.
