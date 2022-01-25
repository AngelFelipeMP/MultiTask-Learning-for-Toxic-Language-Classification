import os
import shutil
import csv
from xmlrpc.client import boolean
import pandas as pd
from sklearn.model_selection import train_test_split

def data_acquisition(source_path=str(), target_folder=str(), datasets=list()):
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
    

def file_list(path, word_in):
    csv_list = os.listdir(path)
    csv_list = [x for x in csv_list if word_in in x]
    print(f'List of {word_in} files/csv {csv_list}')
    return csv_list

# def process_data(file_names, path, text_column, label_column, split_test):
def process_data(path, dataset_name, text_column, label_column):
    file_names = file_list(path, dataset_name)
    dataset_info = dict()
    
    datasets = {'.csv':[name for name in file_names if '.csv' in name],
                '.tsv':[name for name in file_names if '.tsv' in name]}
    
    for k,v in datasets.items():
        divide_columns = ',' if k == '.csv' else '\t'
        for data in v:
            df = pd.read_csv(path + '/' + data, sep=divide_columns)
            
            #remove some "\t" and "\n"
            df[text_column] = df.loc[:,text_column].apply(lambda x: x.replace('\n', ' '))
            df[text_column] = df.loc[:,text_column].apply(lambda x: x.replace('\t', ' '))
            columns_stratify = [label_column, 'language'] if 'language' in df.columns.to_list() else [label_column]
            head = None if type(df.columns.to_list()[0]) != int else True
            
            # save as .tsv
            df.to_csv(path + '/' + data[:-4] + '_processed' + '.tsv', sep="\t")
            
            # save columns to using in the data split for stratification and header None/True
            dataset_info[data[:-4] + '_processed' + '.tsv'] = [columns_stratify, head]
            
    return dataset_info
            


def split_data(data_path=str(), fold_number=int(), test_size=float(), dataset_info=dict()):
    
    for name, info in dataset_info.items():
        df = pd.read_csv(data_path + '/' + name, sep="\t")
            
        if 'train' in name:
            train, val = train_test_split(df, test_size = 0.2, stratify=df.loc[:, info[0][0]], random_state=42)
            train.reset_index(drop=True).to_csv(path + '/' + name[:-4] + '_[TRAIN]' + '.tsv', header=info[1], sep="\t")
            val.reset_index(drop=True).to_csv(path + '/' + name[:-4] + '_[VAL]' + '.tsv', header=info[1], sep="\t")
            
        elif 'test' in name and 'label' in name:
            df.to_csv(path + '/' + name[:-4] + '_[TEST]' + '.tsv', header=info[1], sep="\t")

def change_parameter_seeds(args):
    #### construction
    return

def train(dataset_config=str(), device=int(), output_path=str(), parameter_config=str()):
    code_line = 'python3 train.py --dataset_config ' + dataset_config
    code_line = code_line + ' --device ' + str(device)
    code_line = code_line + ' --name ' + output_path
    code_line = code_line + ' --parameters_config ' + parameter_config
    
    os.system(code_line)

def average(folds_number=int(), log_path=str(), models=list()):
    ''' Average cross validation results and copy all results to local repo '''
    #### construction
    return