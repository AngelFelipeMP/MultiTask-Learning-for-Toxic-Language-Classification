import os
import shutil
import pandas as pd

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
    
    print(code_line)
    # os.system(code_line)



def average(folds_number=int(), log_path=str(), models=list()):
    # TODO -> write avg func
    ''' Average cross validation results and copy all results to local repo '''
    #### construction
    return