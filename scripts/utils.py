import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
    

def file_list(path=str(), word_in=str()):
    file_list = os.listdir(path)
    file_list = [file_name for file_name in file_list if word_in in file_name]
    print(f'List of {word_in} files/csv/tsv {file_list}')
    return file_list


def process_data(path=str(), dataset_name=str(), text_column=str(), label_column=str()):
    file_names = file_list(path, dataset_name)
    dataset_info = dict()
    merge_list = list()
    
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
            data_path_name = path + '/' + data[:-4] + '_processed' + '.tsv'
            df = df.reset_index(drop=True)
            df.to_csv(data_path_name, sep="\t")
            
            # save data train/test to concat
            if v and ('label' in data or 'train' in data):
                merge_list.append(df)
            
            # save columns to using in the data split for stratification and header None/True
            dataset_info[data[:-4] + '_processed' + '.tsv'] = [columns_stratify, head]
            
        # concat train and test
        if merge_list:
            df = pd.concat(merge_list, ignore_index=True)
            df.reset_index(drop=True).to_csv(path + '/' + data.split('_')[0] + '_merge' + '_processed' + '.tsv', sep="\t")
                
    return dataset_info
            


def split_data(data_path=str(), fold_number=int(), test_size=float(), dataset_info=dict()):
    # get rundom_state -> fold_number
    # merge train and test

    
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