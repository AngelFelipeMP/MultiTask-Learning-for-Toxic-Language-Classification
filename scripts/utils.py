import os
import shutil
import pandas as pd
import json
import gdown
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
import numpy as np
import torch


class MtlClass:
    '''Class for train a mtl model'''
    def __init__(self, info_path=str(), fetch_data=True, debug=False):
        self.fetch_data = fetch_data
        self.debug = debug
        self.info_dict = self.read_json('../config' + '/' + info_path)
        self.path = '/' + '/'.join(os.getcwd().split('/')[1:-2])
        self.repo_path = '/' + '/'.join(os.getcwd().split('/')[1:-1])
        self.data_path = self.path + '/data'
        self.config_path = self.repo_path + '/config/' + self.info_dict['experiment']
        self.config_files = self.config_path + '/' + self.info_dict['experiment']
        self.dataset_config = self.config_files + '_data_'
        self.parameter_config = self.config_files + '_parameter_'
        self.process_pre_train()

        
    def process_pre_train(self):
        # dowloand data
        if self.fetch_data and not self.debug:
            self.download_data()
        
        #create a dict with tasks information
        self.get_tasks()
        
        # pytorch check for GPU
        if self.info_dict['device'].lower() == 'auto':
            self.info_dict['device'] = 0 if torch.cuda.is_available() else -1
            
        # process datasets
        for task in self.tasks.keys():
            self.process_data(task, self.tasks[task]['text'])
        
        
        
    def file_list(self, path=str(), word_in=str()):
        # get list with file that contain "word_in" 
        return [file for file in os.listdir(path) if word_in in file]

        
        
    def read_json(self, path=str()):
        # reading info config file
        with open(path, 'r') as f:
            file = f.read()
        return json.loads(file)
            
            
            
    def download_data(self):
        # create a data folder
        if os.path.exists(self.data_path):
            shutil.rmtree(self.data_path)
        os.makedirs(self.data_path)
        
        for task,url in self.info_dict['data_urls'].items():
            #download data folders to current directory
            gdown.download_folder(url, quiet=True)
            sorce_folder = os.getcwd() + '/' + task
            
            # move datasets to the data folder
            file_names = os.listdir(sorce_folder)
            for file_name in file_names:
                shutil.move(os.path.join(sorce_folder, file_name), self.data_path)
                
            # delete data folders from current directory
            shutil.rmtree(sorce_folder)
            
            
            
    def get_tasks(self):
        # get train datasets & mtl config json
        datasets = [dataset for dataset in self.file_list(self.data_path, 'train') if 'processed' not in dataset] 
        self.tasks = dict()

        # open config in python dict
        conf_dict = self.read_json(self.dataset_config + 'mtl_config' + '.json')
        
        # add new information to dict
        for task, info in conf_dict.items():
            self.tasks[task] = dict()
            self.tasks[task]['sent_idxs'] = info['sent_idxs'][0]
            self.tasks[task]['column_idx'] = list(info['tasks'].values())[0]['column_idx']
            self.tasks[task]['train'] = [dataset for dataset in datasets if task in dataset][0]
            self.tasks[task]['split'] = '\t' if '.tsv' in self.tasks[task]['train'] else ','
            
            df = pd.read_csv(self.data_path + '/' + self.tasks[task]['train'], sep=self.tasks[task]['split'])
            
            self.tasks[task]['text'] = list(df.columns)[self.tasks[task]['sent_idxs']-1]
            self.tasks[task]['label'] = list(df.columns)[self.tasks[task]['column_idx']-1]

        
        
    def process_data(self, dataset_name=str(), text_column=str()):
        file_names = self.file_list(self.data_path, dataset_name)
        merge_list = list()
        
        datasets = {'.csv':[name for name in file_names if '.csv' in name and 'processed' not in name],
                    '.tsv':[name for name in file_names if '.tsv' in name and 'processed' not in name]}
        
        # process data for machamp standards &
        for k,v in datasets.items():
            if k:
                divide_columns = ',' if k == '.csv' else '\t'
                for data in v:
                    #DEBUG "nrows" 
                    if self.debug == True:
                        df = pd.read_csv(self.data_path + '/' + data, sep=divide_columns, nrows=32)
                    else:
                        df = pd.read_csv(self.data_path + '/' + data, sep=divide_columns)
                    
                    #remove some "\t" and "\n"
                    df[text_column] = df.loc[:,text_column].apply(lambda x: x.replace('\n', ' '))
                    df[text_column] = df.loc[:,text_column].apply(lambda x: x.replace('\t', ' '))
                    head = None if type(df.columns.to_list()[0]) != int else True
                    
                    # save as .tsv
                    data_path_name = self.data_path + '/' + data[:-4] + '_processed' + '.tsv'
                    df = df.reset_index(drop=True)
                    df.to_csv(data_path_name, header=head, sep="\t")
                    
                    # save data train/test to concat
                    if 'label' in data or 'train' in data:
                        merge_list.append(df)
                            
        # concat train and test
        if merge_list:
            df = pd.concat(merge_list, ignore_index=True)
            df_merged_name = data.split('_')[0] + '_merge' + '_processed' + '.tsv'
            df.reset_index(drop=True).to_csv(self.data_path + '/' + df_merged_name, header=head, sep="\t")

        # if dataset is multilingual get the column index
        language = [df.columns.to_list().index('language')] if 'language' in df.columns.to_list() else []

        # add processe data to task dict
        self.add_to_tasks(dataset_name, language, df_merged_name)
        
        
        
    def add_to_tasks(self, task=str(), lang_index=list(), merged=str()):
        #COMMENT I may need to change kfold if I use only one language from multilingual data
        # create objs to create the k folds
        simple_kfold = StratifiedKFold(n_splits=self.info_dict['folds_number'], random_state=42, shuffle=True)
        multi_label_kfold = MultilabelStratifiedKFold(n_splits=self.info_dict['folds_number'], random_state=42, shuffle=True)
        
        # add info to data/tasks dictionary
        self.tasks[task]['stratify_col'] = [self.tasks[task]['column_idx']] + lang_index
        self.tasks[task]['merged'] = merged
        
        df = pd.read_csv(self.data_path + '/' + merged, header=None, index_col=0, sep="\t").reset_index(drop=True)
        kfold = multi_label_kfold if len(self.tasks[task]['stratify_col']) > 1 else simple_kfold
        self.tasks[task]['df'] = df
        self.tasks[task]['kfold'] = kfold.split(np.zeros(len(df)), df.iloc[:, self.tasks[task]['stratify_col']])
        
        
        
    def train(self, model=str()):
        code_line = 'python train.py --dataset_config ' + self.dataset_config + model + '_config.json'
        code_line = code_line + ' --device ' + str(self.info_dict['device'])
        code_line = code_line + ' --name ' + self.info_dict['experiment'] + '/' + model
        code_line = code_line + ' --parameters_config ' + self.parameter_config + 'config.json'
        
        #DEBUG May only print command
        print('\n')
        print(code_line)
        print('\n')
        os.system(code_line)



def average(folds_number=int(), log_path=str(), models=list()):
    # TODO -> write avg func
    ''' Average cross validation results and copy all results to local repo '''
    #### construction
    return



# def data_acquisition(config_path=str(), source_path=str(), target_folder=str()):
    
#     # get task/dataset names
#     file = file_list(config_path, 'mtl')[0]
#     with open(config_path + '/' + file, 'r') as f:
#         conf_dict = f.read()
#     js = json.loads(conf_dict)
#     datasets = list(js.keys())
    
#     # create a data folder
#     if os.path.exists(target_folder):
#         shutil.rmtree(target_folder)
#     os.makedirs(target_folder)
#     # fetch all files
#     for dataset in datasets:
#         source_folder = source_path + '/' + dataset
#         for file_name in os.listdir(source_folder):
#             # construct full file path
#             source = source_folder + '/' + file_name
#             destination = target_folder + '/' + file_name
#             # copy only files
#             if os.path.isfile(source):
#                 shutil.copy(source, destination)
#                 print('copied', file_name)
