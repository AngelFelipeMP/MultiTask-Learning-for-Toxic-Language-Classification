# from utils import split_data, change_parameter_seeds, train, average
from utils import process_data, data_acquisition
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

folds_number = 5
experiment = 'E1'
models = ['mtl', 'stl_exist', 'stl_detoxis']
device = 0  #gpu 0 / cpu 1
source_data_path = '/content/drive/MyDrive/Code/MTL_2021/Data'
data_path = '/content/data'
path = '/content/drive/MyDrive/Code/MTL_2021/machamp/experiment_one'
repo_path = path + '/MultiTask-Learning-for-Toxic-Language-Classification'
results_path = path + '/E1_results'
config_path = repo_path + '/config/' + experiment
dataset_config = config_path + '_data_'
parameter_config = config_path + '_parameter_'

# if __name__ == '__main__':

# install dependencies
os.system('pip3 install --user -r ' + repo_path + '/machamp/requirements.txt')

# grab data from drive
data_acquisition(source_data_path, data_path, ['EXIST','DETOXIS'])

# process data for machamp standards
process_data(data_path, 'DETOXIS', 'comment', 'toxicity')
process_data(data_path, 'EXIST', 'text', 'task1')

#TODO comment the code below
#TODO check if the code are smaller as possible
#TODO check if it is batter create a function 
# create skf object to stratify the data split
detoxis_file = 'DETOXIS2021_merge_processed.tsv'
exist_file = 'EXIST2021_merge_processed.tsv'
#TODO get the dataset name from process_data func
df_detoxis = pd.read_csv(data_path + '/' + detoxis_file, header=None, index_col=0, sep="\t").reset_index(drop=True)
df_exist = pd.read_csv(data_path + '/' + exist_file, header=None, index_col=0, sep="\t").reset_index(drop=True)

kfold = StratifiedKFold(n_splits=folds_number, random_state=42, shuffle=True)
strat_kfold = MultilabelStratifiedKFold(n_splits=folds_number, random_state=42, shuffle=True)

for data_exist, data_detoxis in zip(strat_kfold.split(np.zeros(len(df_exist)), df_exist.iloc[:,[3,5]]), kfold.split(np.zeros(len(df_detoxis)), df_detoxis.iloc[:,[19]])):
    
    df_detoxis.iloc[data_detoxis[0]].reset_index(drop=True).to_csv(data_path + '/' + detoxis_file.split('_')[0] + '_processed' + '_[TRAIN]' + '.tsv', header=None, sep="\t") 
    df_detoxis.iloc[data_detoxis[1]].reset_index(drop=True).to_csv(data_path + '/' + detoxis_file.split('_')[0] + '_processed' + '_[VAL]' + '.tsv', header=None, sep="\t")
    
    df_exist.iloc[data_exist[0]].reset_index(drop=True).to_csv(data_path + '/' + exist_file.split('_')[0] + '_processed' + '_[TRAIN]' + '.tsv', header=None, sep="\t") 
    df_exist.iloc[data_exist[1]].reset_index(drop=True).to_csv(data_path + '/' + exist_file.split('_')[0] + '_processed' + '_[VAL]' + '.tsv', header=None, sep="\t")

    print(data_exist)
    print(data_detoxis)
    print('#######################')
    
    #TODO add the below part to the code
    # split_data(data_path, train_index, test_index, 0.2, stratify_head_exist)

    # change_parameter_seeds(parameter_config + 'config.json')
    # for model in models:
    #     output_path = experiment + '/' + model 
    #     train(
    #         dataset_config + model + '_config.json',
    #         device,
    #         output_path,
    #         parameter_config + 'config.json')


#TODO add the avg func
# average(folds_number, repo_path + '/machamp/logs/' + experiment, models)