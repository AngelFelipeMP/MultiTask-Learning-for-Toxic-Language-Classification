# dependencies
import argparse
from distutils.log import debug
import os
import pandas as pd
import numpy as np
import torch
import json
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from utils import process_data, data_acquisition, train, get_tasks, download_data

parser = argparse.ArgumentParser()
parser.add_argument("--information_config", default="", type=str, help="Modes configuration file")
parser.add_argument("--debug", default=False, help="Must be True or False", action='store_true')
args = parser.parse_args()

if args.information_config == '':
    print('Specifying --information_config path is required')
    exit(1)

print('****************************')
print(args.debug)
print(type(args.debug))
print('****************************')
    
# reading info config file
with open('../config' + '/' + args.information_config, 'r') as f:
    conf_info_dict = f.read()
info = json.loads(conf_info_dict)

# base data/config/results path
path = '/' + '/'.join(os.path.abspath(os.getcwd()).split('/')[1:-2])
repo_path = '/' + '/'.join(os.path.abspath(os.getcwd()).split('/')[1:-1])
data_path = path + '/data'
results_path = path + '/E1_results'
config_path = repo_path + '/config/' + info['experiment']
config_files = config_path + '/' + info['experiment']
info_config = config_files + '_informatio_'
dataset_config = config_files + '_data_'
parameter_config = config_files + '_parameter_'

# grab data from drive #DEBUG download data
if args.debug == False:
    download_data(info['data_urls'], data_path)

print()
# TODO finish get_task func and modify process_data func at the same time
tasks = get_tasks(info['experiment'], config_path, data_path)

#pytorch check for GPU
if info['device'].lower() == 'auto':
    info['device'] = 0 if torch.cuda.is_available() else -1
    
# create objs to create the k folds
simple_kfold = StratifiedKFold(n_splits=info['folds_number'], random_state=42, shuffle=True)
multi_label_kfold = MultilabelStratifiedKFold(n_splits=info['folds_number'], random_state=42, shuffle=True)

# process data for machamp standards & add info to data/taks dictionary
for task in tasks.keys():
    file, lang_index = process_data(data_path, task, tasks[task]['text'], args.debug)
    tasks[task]['stratify_col'] = [tasks[task]['column_idx']] + lang_index
    tasks[task]['file'] = file
    
    df = pd.read_csv(data_path + '/' + file, header=None, index_col=0, sep="\t").reset_index(drop=True)
    kfold = multi_label_kfold if len(tasks[task]['stratify_col']) > 1 else simple_kfold
    tasks[task]['df'] = df
    tasks[task]['kfold'] = kfold.split(np.zeros(len(df)), df.iloc[:, tasks[task]['stratify_col']])

# Save data folds and Train the models
split_sequence = tasks.keys()
for idxs in zip(*[tasks[data]['kfold'] for data in split_sequence]):
    for idx,task in zip(idxs, split_sequence):
        tasks[task]['df'].iloc[idx[0]].reset_index(drop=True).to_csv(data_path + '/' + tasks[task]['file'].split('_')[0] + '_processed' + '_[TRAIN]' + '.tsv', header=None, sep="\t")
        tasks[task]['df'].iloc[idx[1]].reset_index(drop=True).to_csv(data_path + '/' + tasks[task]['file'].split('_')[0] + '_processed' + '_[VAL]' + '.tsv', header=None, sep="\t")
    
    for model in ['mtl'] + ['stl_' + tasks.lower() for tasks in tasks.keys()]:
        output_path = info['experiment'] + '/' + model 
        train(
            dataset_config + model + '_config.json',
            info['device'],
            output_path,
            parameter_config + 'config.json')
            
        #DEBUG add "Break"
        if args.debug == True:
            break
    
    #DEBUG add "Break"
    if args.debug == True:
        break

# TODO group all functions in a class

# TODO change debug for a away to test the code outomatic

#TODO add the avg func
# average(info['folds_number'], repo_path + '/machamp/logs/' + info['experiment'], models)
