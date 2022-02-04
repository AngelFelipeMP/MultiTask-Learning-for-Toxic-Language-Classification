# dependencies
import argparse
import os
import pandas as pd
import numpy as np
import torch
import json
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from utils import process_data, data_acquisition, train, get_tasks

parser = argparse.ArgumentParser()
parser.add_argument("--information_config", default="", type=str, help="Modes configuration file")
args = parser.parse_args()

if args.information_config == '':
    logger.error('Specifying --information_config path is required')
    exit(1)


#TODO Criate a config file to add out the path stuff
#TODO Organaze code

# reading info config file
with open('../config' + '/' + args.information_config, 'r') as f:  # "/E1/E1_information_config.json"
    conf_dict = f.read()
info = json.loads(conf_dict)

#inputs
# folds_number = 2 #at least 2 (Train/val)
# experiment = 'E1'
# source_data_path = '/Users/angel_de_paula/angel.magnossao@alumni.usp.br - Google Drive/My Drive/Code/MTL_2021/Data'
# path = '/Users/angel_de_paula/repos/mtl' #main path

path = '/' + '/'.join(os.path.abspath(os.getcwd()).split('/')[1:-2])
repo_path = '/' + '/'.join(os.path.abspath(os.getcwd()).split('/')[1:-1])
data_path = path + '/data'
results_path = path + '/E1_results'
config_path = repo_path + '/config/' + info['experiment']
config_files = config_path + '/' + info['experiment']
info_config = config_files + '_informatio_'
dataset_config = config_files + '_data_'
parameter_config = config_files + '_parameter_'

# TODO Change data_aquisition to download data from drive
# grab data from drive
data_acquisition(config_path, info['source_data_path'], data_path)

get_tasks(info['experiment'], config_path, data_path)
tasks = {'DETOXIS':{'text':'comment','label':'toxicity'},
            'EXIST':{'text':'text','label':'task1'}}

#pytorch check for GPU
if info['device'].lower() == 'auto':
    info['device'] = 0 if torch.cuda.is_available() else -1
    
# create objs to create the k folds
simple_kfold = StratifiedKFold(n_splits=info['folds_number'], random_state=42, shuffle=True)
multi_label_kfold = MultilabelStratifiedKFold(n_splits=info['folds_number'], random_state=42, shuffle=True)

#TODO test the code
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
        output_path = info['experiment'] + '/' + model 
        train(
            dataset_config + model + '_config.json',
            info['device'],
            output_path,
            parameter_config + 'config.json')
            
        #DEBUG add "Break" for the BUG purpose
        break
    
    #DEBUG add "Break" for the BUG purpose
    break

# TODO group all functions in a class

# TODO change debug for a away to test the code outomatic

#TODO add the avg func
# average(info['folds_number'], repo_path + '/machamp/logs/' + info['experiment'], models)

