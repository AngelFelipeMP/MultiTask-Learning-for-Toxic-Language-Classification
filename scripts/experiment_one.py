#TODO Criate a config file to add out the path stuff
# # inputs
# folds_number = 2 #at least 2 (Train/val)
# experiment = 'E1'
# device = 0  #gpu 0 / cpu -1
# source_data_path = '/content/drive/MyDrive/Code/MTL_2021/Data'
# data_path = '/content/data'
# path = '/content' #main path
# repo_path = path + '/MultiTask-Learning-for-Toxic-Language-Classification'
# results_path = path + '/E1_results'
# config_path = repo_path + '/config/' + experiment
# dataset_config = config_path + '/' + experiment + '_data_'
# parameter_config = config_path + '/' + experiment + '_parameter_'

# # inputs
# folds_number = 2 #at least 2 (Train/val)
# experiment = 'E1'
# device = -1  #gpu 0 / cpu -1
# source_data_path = '/home/angel/uspdrive/Code/MTL_2021/Data'
# data_path = '/home/angel/repos/mtl/data'
# path = '/home/angel/repos/mtl' #main path
# repo_path = path + '/MultiTask-Learning-for-Toxic-Language-Classification'
# results_path = path + '/E1_results'
# config_path = repo_path + '/config/' + experiment
# dataset_config = config_path + '/' + experiment + '_data_'
# parameter_config = config_path + '/' + experiment + '_parameter_'

# inputs
folds_number = 2 #at least 2 (Train/val)
experiment = 'E1'
device = None
source_data_path = '/Users/angel_de_paula/angel.magnossao@alumni.usp.br - Google Drive/My Drive/Code/MTL_2021/Data'
path = '/Users/angel_de_paula/repos/mtl' #main path
data_path = path + '/data'
repo_path = path + '/MultiTask-Learning-for-Toxic-Language-Classification'
results_path = path + '/E1_results'
config_path = repo_path + '/config/' + experiment
dataset_config = config_path + '/' + experiment + '_data_'
parameter_config = config_path + '/' + experiment + '_parameter_'

from ast import Break
from utils import process_data, data_acquisition, train, get_tasks
import torch

# TODO Change data_aquisition to download data from drive
# grab data from drive
data_acquisition(config_path, source_data_path, data_path)

get_tasks(experiment, config_path, data_path)
tasks = {'DETOXIS':{'text':'comment','label':'toxicity'},
            'EXIST':{'text':'text','label':'task1'}}
#TEST

# install dependencies & import
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

#pytorch check for GPU
if device == str() and device.lower() == 'auto':
    device = 0 if torch.cuda.is_available() else -1

# create objs to create the k folds
simple_kfold = StratifiedKFold(n_splits=folds_number, random_state=42, shuffle=True)
multi_label_kfold = MultilabelStratifiedKFold(n_splits=folds_number, random_state=42, shuffle=True)

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
        output_path = experiment + '/' + model 
        train(
            dataset_config + model + '_config.json',
            device,
            output_path,
            parameter_config + 'config.json')
            
        #DEBUG add "Break" for the BUG purpose
        break

# TODO group all functions in a class

# TODO change debug for a away to test the code outomatic

#TODO add the avg func
# average(folds_number, repo_path + '/machamp/logs/' + experiment, models)

