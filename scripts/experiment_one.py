# from utils import split_data, change_parameter_seeds, train, average
from utils import process_data, data_acquisition
import os

folds_number = 1
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

# grab data from drive
data_acquisition(source_data_path, data_path, ['EXIST','DETOXIS'])

stratify_head_detoxis = process_data(data_path, 'DETOXIS', 'comment', 'toxicity')
stratify_head_exist = process_data(data_path, 'EXIST', 'text', 'task1')

print(stratify_head_exist)
print(stratify_head_detoxis)

# for fold in range(folds_number):
#     split_data(data_path, fold, 0.2, stratify_head_detoxis)
#     split_data(data_path, fold, 0.2, stratify_head_exist)
    
#     change_parameter_seeds(parameter_config + 'config.json')
#     for model in models:
#         output_path = experiment + '/' + model 
#         train(
#             dataset_config + model + '_config.json',
#             device,
#             output_path,
#             parameter_config + 'config.json')

# average(folds_number, repo_path + '/machamp/logs/' + experiment, models)

## //TODO
# write the data_split script fuction (utils.py)
# write the change_parameter_seeds fuction (utils.py)
# debug process_data
# debug split_data
# add randon seed split data
##### write the fuctions by hand first !!!!!! ########