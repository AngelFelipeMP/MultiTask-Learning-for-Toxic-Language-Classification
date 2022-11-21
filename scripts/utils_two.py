from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import shutil
import pandas as pd
import json
import gdown
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
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
        self.logs_path = self.repo_path + '/machamp/logs/' + self.info_dict['experiment']
        self.config_path = self.repo_path + '/config/' + self.info_dict['experiment']
        self.config_files = self.config_path + '/' + self.info_dict['experiment']
        self.dataset_config = self.config_files + '_data_'
        self.parameter_config = self.config_files + '_parameter_'
        self.process_pre_train()
        self.models()

    def models(self):
        return ['_'.join(file.split('_')[2:4]) for file in os.listdir(self.config_path) if 'mtl'in file or 'stl' in file]
        
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
        file_list = [file for file in os.listdir(self.config_path) if 'stl' in file]
        for file in file_list:
            conf_dict = self.read_json(self.config_path + '/' + file)
        
            # add new information to dict
            for task, info in conf_dict.items():
                self.tasks[task] = dict()
                self.tasks[task]['metric'] = list(info['tasks'].values())[0]['metric']
                self.tasks[task]['sent_idxs'] = info['sent_idxs'][0]
                self.tasks[task]['column_idx'] = list(info['tasks'].values())[0]['column_idx']
                self.tasks[task]['train'] = [dataset for dataset in datasets if task in dataset][0]
                self.tasks[task]['split'] = '\t' if '.tsv' in self.tasks[task]['train'] else ','
            
                df = pd.read_csv(self.data_path + '/' + self.tasks[task]['train'], sep=self.tasks[task]['split'])
            
                self.tasks[task]['text'] = list(df.columns)[self.tasks[task]['sent_idxs']]
                self.tasks[task]['label'] = list(df.columns)[self.tasks[task]['column_idx']]
            
            
    def process_data(self, dataset_name=str(), text_column=str()):
        file_names = self.file_list(self.data_path, dataset_name)
        merge_list = list()
        
        datasets = {'.csv':[name for name in file_names if '.csv' in name and 'processed' not in name],
                    '.tsv':[name for name in file_names if '.tsv' in name and 'processed' not in name]}
        
        # process data for machamp standards
        for k,v in datasets.items():
            if k:
                divide_columns = ',' if k == '.csv' else '\t'
                for data in v:
                    #DEBUG "nrows" 
                    if self.debug:
                        df = pd.read_csv(self.data_path + '/' + data, sep=divide_columns, nrows=32)
                    else:
                        df = pd.read_csv(self.data_path + '/' + data, sep=divide_columns)
                    
                    #remove non-target language text
                    if self.info_dict['language'] and 'language' in df.columns.to_list():
                        df = df.loc[df['language'] == self.info_dict['language']].reset_index(drop=True)
                        df.to_csv(self.data_path + '/' + data, index=False, sep=divide_columns)
                    
                    #remove some "\t" and "\n"
                    df[text_column] = df.loc[:,text_column].apply(lambda x: x.replace('\n', ' '))
                    df[text_column] = df.loc[:,text_column].apply(lambda x: x.replace('\t', ' '))
                    head = None if type(df.columns.to_list()[-1]) != int else True
                    
                    # save as .tsv
                    data_path_name = self.data_path + '/' + data[:-4] + '_processed' + '.tsv'
                    df.to_csv(data_path_name, index=False, header=head, sep="\t")
                    
                    # save data train/test to concat
                    if 'label' in data or 'train' in data:
                        merge_list.append(df)
                            
        # concat train and test
        if merge_list:
            df = pd.concat(merge_list, ignore_index=True)
            
            # remove duplicate instances
            df.drop_duplicates(text_column, inplace=True)
            
            # save as .tsv
            df_merged_name = data.split('_')[0] + '_merge' + '_processed' + '.tsv'
            df.to_csv(self.data_path + '/' + df_merged_name, header=head, index=False, sep="\t")

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
        
        df = pd.read_csv(self.data_path + '/' + merged, header=None, sep="\t").reset_index(drop=True)
        
        kfold = multi_label_kfold if len(self.tasks[task]['stratify_col']) > 1 else simple_kfold
        self.tasks[task]['df'] = df
        self.tasks[task]['kfold'] = kfold.split(np.zeros(len(df)), df.iloc[:, self.tasks[task]['stratify_col']])
        
        
    def train(self, model=str()):
        code_line = 'python train.py --dataset_config ' + self.dataset_config + model + '_config.json'
        code_line = code_line + ' --device ' + str(self.info_dict['device'])
        code_line = code_line + ' --name ' + self.info_dict['experiment'] + '/' + model
        code_line = code_line + ' --parameters_config ' + self.parameter_config + 'config.json'
        
        print('\n ' + code_line + '\n')
        os.system(code_line)
        
        
        
    def time_str_to_float(self, time=str()):
        '''' time from "h:m:s" (str) to seconds "s" (float) '''
        time = [ float(t) for t in time.split(':')]
        time = (time[0]*60)*60 + time[1]*60 + time[2]
        return time
    
    def seg_to_time(self,time=float()):
        '''' time from seconds "s" to "h:m:s" '''
        hours = int(time // 3600)
        minutes = int((time % 3600) // 60)
        segundos = round((time % 3600) % 60, 6)
        return str(hours) + ':' + str(minutes) + ':' + str(segundos) 

    def average(self):
        ''' Average cross validation results and copy all results to local repo '''
        
        for model_result in os.listdir(self.logs_path):
            # list to save model results
            list_results=list()
            
            # get score using the prediction in each fold
            average_score = self.score(model_result)
            
            for fold_result  in os.listdir(self.logs_path + '/' + model_result):
                if 'predictions' not in fold_result and 'average' not in fold_result: 
                
                    results = self.read_json(path = self.logs_path + '/' + model_result + '/' + fold_result + '/metrics' + '.json')
                    results["training_duration"] = self.time_str_to_float(results["training_duration"])
                    list_results.append(results)
            
            average_results = dict(pd.DataFrame(list_results).mean())
            # return some columns to int
            average_results['training_start_epoch'] = int(average_results['training_start_epoch'])
            average_results['training_epochs'] = int(average_results['training_epochs'])
            average_results['epoch'] = int(average_results['epoch'])
            
            average_results['training_duration'] = self.seg_to_time(average_results['training_duration'])
            
            # add score caculation using all data
            for k,v in average_score.items():
                average_results[k] = v
            
            # save average results as .json
            with open(self.logs_path + '/' + model_result + '/' + 'average.json', 'w') as fp:
                json.dump(average_results, fp, indent=2)
        
        
        
    def score(self, model_result):
        data_info = {task:{'file':task + '.dev.out', 'data':[]} for task in self.tasks.keys()}
        
        # aggregating data/predictions
        for fold_result  in os.listdir(self.logs_path + '/' + model_result):
            if 'predictions' not in fold_result and 'average' not in fold_result: 

                for task in data_info.keys():
                    if os.path.exists(self.logs_path + '/' + model_result + '/' + fold_result + '/' + data_info[task]['file']):
                    
                        data_info[task]['data'].append(pd.read_csv(self.logs_path + '/' + model_result + '/' + fold_result + '/' + data_info[task]['file'], 
                                                        header=None, 
                                                        sep="\t"))
        # save predictionds 
        results_dict = dict()
        for task in data_info.keys():
            if data_info[task]['data']:
                
                # merge cv predictions   
                df_predict = pd.concat(data_info[task]['data'], ignore_index=True).iloc[:,[self.tasks[task]['sent_idxs'],self.tasks[task]['column_idx']]]
                
                # read cv dataset               
                df_merge = pd.read_csv(self.data_path + '/' + self.tasks[task]['merged'], header=None, sep="\t").reset_index(drop=True)
                
                # combay prediction & labels
                df_predict_merge = pd.merge(df_merge, df_predict, left_on=self.tasks[task]['sent_idxs'], right_on=self.tasks[task]['sent_idxs'])
                
                # save predions + labels data
                df_predict_merge.to_csv(self.logs_path + '/' + model_result + '/' + task + '_predictions' + '.tsv', index=False, header=None, sep="\t")
                
                # calculate scores
                if self.tasks[task]['metric'] == 'acc':
                    score = accuracy_score(df_predict_merge.iloc[:,self.tasks[task]['column_idx']], df_predict_merge.iloc[:,-1])
                # f1 score - not averaged
                elif 'f1_' in self.tasks[task]['metric']:
                    score = f1_score(df_predict_merge.iloc[:,self.tasks[task]['column_idx']].to_list(), df_predict_merge.iloc[:,-1].to_list(),pos_label=self.tasks[task]['metric'].split('_')[1])
                elif '-f1' in self.tasks[task]['metric']:
                    score = f1_score(df_predict_merge.iloc[:,self.tasks[task]['column_idx']].to_list(), df_predict_merge.iloc[:,-1].to_list(), average=self.tasks[task]['metric'].split('-')[0])
                
                results_dict[task + '_crossvalidation_' + self.tasks[task]['metric']] = score                
                
        return results_dict
    
    def upload_data(self):
        
        GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = self.repo_path + '/config' + '/' + 'client_secrets.json'
        
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile(self.repo_path + '/config' + '/' + 'mycreds.txt')
        
        if gauth.credentials is None:
            # Authenticate if they're not there
            gauth.LocalWebserverAuth()
        
        elif gauth.access_token_expired:
            # Refresh them if expired
            gauth.Refresh()
        
        else:
            # Initialize the saved creds
            gauth.Authorize()
        # Save the current credentials to a file
        gauth.SaveCredentialsFile(self.repo_path + '/config'+ '/' + 'mycreds.txt')
        
        drive = GoogleDrive(gauth)
        shutil.make_archive(self.logs_path + '/' + 'results', 'zip', self.logs_path)
        file_drive = drive.CreateFile({'parents': [{'id': '1o8ZHptI_J-0jP8PGdIKB3CbVrUtj3r-G'}]})
        file_drive.SetContentFile(self.logs_path + '/' +'results.zip')
        file_drive.Upload()