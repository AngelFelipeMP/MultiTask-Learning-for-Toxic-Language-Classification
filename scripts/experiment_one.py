# inputs
folds_number = 2 #at least 2 (Train/val)
experiment = 'E1'
device = 0  #gpu 0 / cpu 1
source_data_path = '/content/drive/MyDrive/Code/MTL_2021/Data'
data_path = '/content/data'
path = '/content/drive/MyDrive/Code/MTL_2021/machamp/experiment_one'
repo_path = path + '/MultiTask-Learning-for-Toxic-Language-Classification'
results_path = path + '/E1_results'
config_path = repo_path + '/config/' + experiment
dataset_config = config_path + '_data_'
parameter_config = config_path + '_parameter_'

tasks = {'DETOXIS':{'text':'comment','label':'toxicity'},
            'EXIST':{'text':'text','label':'task1'}}

# install dependencies & import
import os
# os.system('pip install --user -r ' + repo_path + '/requirements.txt')
os.system('pip install -v -q iterative-stratification')
os.system('pip3 install --user -r ' + repo_path + '/machamp/requirements.txt')
import time
from utils import process_data, data_acquisition, train
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold 

# if __name__ == '__main__':

# grab data from drive
data_acquisition(source_data_path, data_path, tasks.keys())

# create objs to create the k folds
simple_kfold = StratifiedKFold(n_splits=folds_number, random_state=42, shuffle=True)
multi_label_kfold = MultilabelStratifiedKFold(n_splits=folds_number, random_state=42, shuffle=True)

#TODO test the code
#TODO may comment the code abouve
# process data for machamp standards & add info to data/taks dictionary
for task in tasks.keys():
    file, stratify_col = process_data(data_path, task, tasks[task]['text'], tasks[task]['label'])
    df = pd.read_csv(data_path + '/' + file, header=None, index_col=0, sep="\t").reset_index(drop=True)
    kfold = multi_label_kfold if len(stratify_col) > 1 else simple_kfold
    
    tasks[task]['file'] = file
    tasks[task]['stratify_col'] = stratify_col
    tasks[task]['df'] = df
    tasks[task]['kfold'] = kfold.split(np.zeros(len(df)), df.iloc[:, stratify_col])

# Save data folds and Train the models
split_sequence = tasks.keys()
for idxs in zip(*[tasks[data]['kfold'] for data in split_sequence]):
    for idx,task in zip(idxs, split_sequence):
        tasks[task]['df'].iloc[idx[0]].reset_index(drop=True).to_csv(data_path + '/' + tasks[task]['file'].split('_')[0] + '_processed' + '_[TRAIN]' + '.tsv', header=None, sep="\t")
        tasks[task]['df'].iloc[idx[1]].reset_index(drop=True).to_csv(data_path + '/' + tasks[task]['file'].split('_')[0] + '_processed' + '_[VAL]' + '.tsv', header=None, sep="\t")
    
    for model in ['mtl'] + ['stl_' + tasks.lower() for tasks in tasks.keys()]:
        output_path = experiment + '/' + model 
        train(
            dataset_config + model + '_config.json',
            device,
            output_path,
            parameter_config + 'config.json')



#TODO add the avg func
# average(folds_number, repo_path + '/machamp/logs/' + experiment, models)
