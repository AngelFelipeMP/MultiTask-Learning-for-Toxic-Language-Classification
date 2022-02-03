import os
import shutil
import pandas as pd
import json

def data_acquisition(config_path=str(), source_path=str(), target_folder=str()):
    
    # get task/dataset names
    print(config_path)
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
    print(f'List of {word_in} files/csv/tsv {file_list}')
    return file_list



def process_data(path=str(), dataset_name=str(), text_column=str(), label_column=str()):
    file_names = file_list(path, dataset_name)
    merge_list = list()
    stratify_index = list()
    
    datasets = {'.csv':[name for name in file_names if '.csv' in name],
                '.tsv':[name for name in file_names if '.tsv' in name]}
    
    for k,v in datasets.items():
        divide_columns = ',' if k == '.csv' else '\t'
        for data in v:
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
#COMMENT I may can remove it (BELOW)
    # Get df column index for extratification
    if 'language' in df.columns.to_list():
        stratify_index.append(df.columns.to_list().index('language'))
        stratify_index.append(df.columns.to_list().index(label_column))
    else:
        stratify_index.append(df.columns.to_list().index(label_column))

    return df_merged_name, stratify_index



def train(dataset_config=str(), device=int(), output_path=str(), parameter_config=str()):
    code_line = 'python3 train.py --dataset_config ' + dataset_config
    code_line = code_line + ' --device ' + str(device)
    code_line = code_line + ' --name ' + output_path
    code_line = code_line + ' --parameters_config ' + parameter_config
    
    # print(code_line)
    os.system(code_line)

# TODO finish function
def get_tasks(experiment=str(), path=str(), data_path=str()):
    datasets = file_list(data_path, 'train')
    datasets = [dataset for dataset in datasets if 'processed' not in dataset]
    print(datasets)
    print('##################################')
    
    file = file_list(path, 'mtl')[0]
    tasks = dict()

    with open(path + '/' + file, 'r') as f:
        conf_dict = f.read()
        
    js = json.loads(conf_dict)
    
    print(js)
    print('##################################')
    for task, info in js.items():
        tasks[task] = dict()
        tasks[task]['sent_idxs'] = info['sent_idxs'][0]
        tasks[task]['column_idx'] = list(info['tasks'].values())[0]['column_idx']
        tasks[task]['train'] = [dataset for dataset in datasets if task in dataset][0]
        tasks[task]['split'] = '\t' if '.tsv' in tasks[task]['train'] else ','
        
        df = pd.read_csv(data_path + '/' + tasks[task]['train'], sep=tasks[task]['split'])
        print(df.head())
        print('##################################')
        print(tasks[task])
        print('##################################')
    
    print(js)
    print('##################################')

    return 

def average(folds_number=int(), log_path=str(), models=list()):
    # TODO -> write avg func
    ''' Average cross validation results and copy all results to local repo '''
    #### construction
    return