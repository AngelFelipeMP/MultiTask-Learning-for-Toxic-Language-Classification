import os
import shutil
import pandas as pd
import json
import gdown


class MtlClass:
    '''Class for train a mtl model'''
    def __init__(self, info=dict()):
        self.info_dict = info
        self.path = '/' + '/'.join(os.path.abspath(os.getcwd()).split('/')[1:-2])
        self.repo_path = '/' + '/'.join(os.path.abspath(os.getcwd()).split('/')[1:-1])
        self.data_path = self.path + '/data'
        self.results_path = self.path + '/E1_results'
        self.config_path = self.repo_path + '/config/' + self.info_dict['experiment']
        self.config_files = self.config_path + '/' + self.info_dict['experiment']
        self.info_config = self.config_files + '_informatio_'
        self.dataset_config = self.config_files + '_data_'
        self.parameter_config = self.config_files + '_parameter_'
        
        
    def download_data(self):
        # create a data folder
        if os.path.exists(self.data_path):
            shutil.rmtree(self.data_path)
        os.makedirs(self.data_path)
        
        for task,url in self.info_dict['data_urls'].items():
            #download data folders to current directory
            gdown.download_folder(url, quiet=True)
            sorce_folder = os.path.abspath(os.getcwd()) + '/' + task
            
            # move datasets to the data folder
            file_names = os.listdir(sorce_folder)
            for file_name in file_names:
                shutil.move(os.path.join(sorce_folder, file_name), self.data_path)
                
            # delete data folders from current directory
            shutil.rmtree(sorce_folder)
            
            
    def get_tasks(self):
        # get train datasets & mtl config json
        datasets = file_list(self.data_path, 'train')
        config_json = file_list(self.config_path, 'mtl')[0]
        tasks = dict()

        # open config in python dict
        with open(self.config_path + '/' + config_json, 'r') as f:
            conf_dict = f.read()        
        conf_dict = json.loads(conf_dict)
        
        # add new information to dict
        for task, info in conf_dict.items():
            tasks[task] = dict()
            tasks[task]['sent_idxs'] = info['sent_idxs'][0]
            tasks[task]['column_idx'] = list(info['tasks'].values())[0]['column_idx']
            tasks[task]['train'] = [dataset for dataset in datasets if task in dataset and 'processed' not in dataset][0]
            tasks[task]['split'] = '\t' if '.tsv' in tasks[task]['train'] else ','
            
            df = pd.read_csv(self.data_path + '/' + tasks[task]['train'], sep=tasks[task]['split'])
            
            tasks[task]['text'] = list(df.columns)[tasks[task]['sent_idxs']-1]
            tasks[task]['label'] = list(df.columns)[tasks[task]['column_idx']-1]
            
            # save tasks for the all class
            self.tasks = tasks
            
        return self.tasks



# def download_data(urls=dict(), target_folder=str()):
    
#     # create a data folder
#     if os.path.exists(target_folder):
#         shutil.rmtree(target_folder)
#     os.makedirs(target_folder)
    
#     for task,url in urls.items():
#         #download data folders to current directory
#         gdown.download_folder(url, quiet=True)
#         sorce_folder = os.path.abspath(os.getcwd()) + '/' + task
        
#         # move datasets to the data folder
#         file_names = os.listdir(sorce_folder)
#         for file_name in file_names:
#             shutil.move(os.path.join(sorce_folder, file_name), target_folder)
            
#         # delete data folders from current directory
#         shutil.rmtree(sorce_folder)


# def get_tasks(config_path=str(), data_path=str()):
    
#     # get train datasets & mtl config json
#     datasets = file_list(data_path, 'train')
#     config_json = file_list(config_path, 'mtl')[0]
#     tasks = dict()

#     # open config in python dict
#     with open(config_path + '/' + config_json, 'r') as f:
#         conf_dict = f.read()        
#     conf_dict = json.loads(conf_dict)
    
#     # add new information to dict
#     for task, info in conf_dict.items():
#         tasks[task] = dict()
#         tasks[task]['sent_idxs'] = info['sent_idxs'][0]
#         tasks[task]['column_idx'] = list(info['tasks'].values())[0]['column_idx']
#         tasks[task]['train'] = [dataset for dataset in datasets if task in dataset and 'processed' not in dataset][0]
#         tasks[task]['split'] = '\t' if '.tsv' in tasks[task]['train'] else ','
        
#         df = pd.read_csv(data_path + '/' + tasks[task]['train'], sep=tasks[task]['split'])
        
#         tasks[task]['text'] = list(df.columns)[tasks[task]['sent_idxs']-1]
#         tasks[task]['label'] = list(df.columns)[tasks[task]['column_idx']-1]
        
#     return tasks




def process_data(path=str(), dataset_name=str(), text_column=str(), debug=bool()):
    
    file_names = file_list(path, dataset_name)
    merge_list = list()
    
    datasets = {'.csv':[name for name in file_names if '.csv' in name and 'processed' not in name],
                '.tsv':[name for name in file_names if '.tsv' in name and 'processed' not in name]}
    
    for k,v in datasets.items():
        divide_columns = ',' if k == '.csv' else '\t'
        for data in v:
            #DEBUG "nrows" 
            if debug == True:
                df = pd.read_csv(path + '/' + data, sep=divide_columns, nrows=32)
            else:
                df = pd.read_csv(path + '/' + data, sep=divide_columns)
            
            #remove some "\t" and "\n"
            df[text_column] = df.loc[:,text_column].apply(lambda x: x.replace('\n', ' '))
            df[text_column] = df.loc[:,text_column].apply(lambda x: x.replace('\t', ' '))
            head = None if type(df.columns.to_list()[0]) != int else True
            
            # save as .tsv
            data_path_name = path + '/' + data[:-4] + '_processed' + '.tsv'
            df = df.reset_index(drop=True)
            df.to_csv(data_path_name, header=head, sep="\t")
            
            # save data train/test to concat
            if 'label' in data or 'train' in data:
                merge_list.append(df)
                        
    # concat train and test
    if merge_list:
        df = pd.concat(merge_list, ignore_index=True)
        df_merged_name = data.split('_')[0] + '_merge' + '_processed' + '.tsv'
        df.reset_index(drop=True).to_csv(path + '/' + df_merged_name, header=head, sep="\t")

    # if dataset is multilingual get the column index
    language = [df.columns.to_list().index('language')] if 'language' in df.columns.to_list() else []

    return df_merged_name, language



def train(dataset_config=str(), device=int(), output_path=str(), parameter_config=str()):
    
    code_line = 'python3 train.py --dataset_config ' + dataset_config
    code_line = code_line + ' --device ' + str(device)
    code_line = code_line + ' --name ' + output_path
    code_line = code_line + ' --parameters_config ' + parameter_config
    
    #DEBUG May only print command
    print('\n')
    print(code_line)
    print('\n')
    # os.system(code_line)



def data_acquisition(config_path=str(), source_path=str(), target_folder=str()):
    
    # get task/dataset names
    file = file_list(config_path, 'mtl')[0]
    with open(config_path + '/' + file, 'r') as f:
        conf_dict = f.read()
    js = json.loads(conf_dict)
    datasets = list(js.keys())
    
    # create a data folder
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)
    # fetch all files
    for dataset in datasets:
        source_folder = source_path + '/' + dataset
        for file_name in os.listdir(source_folder):
            # construct full file path
            source = source_folder + '/' + file_name
            destination = target_folder + '/' + file_name
            # copy only files
            if os.path.isfile(source):
                shutil.copy(source, destination)
                print('copied', file_name)



def file_list(path=str(), word_in=str()):
    
    file_list = os.listdir(path)
    file_list = [file_name for file_name in file_list if word_in in file_name]
    # spprint(f'List of {word_in} files {file_list}')
    return file_list



def average(folds_number=int(), log_path=str(), models=list()):
    # TODO -> write avg func
    ''' Average cross validation results and copy all results to local repo '''
    #### construction
    return