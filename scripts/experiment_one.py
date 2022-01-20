from utils import split_data, change_parameter_seeds, train

folds_number = 1
experiment = 'E1'
models = ['mtl', 'stl_exist', 'stl_detoxis']
device = 0  #gpu 0 / cpu 1
data_path = '/data'
path = '/content/drive/MyDrive/Code/MTL_2021/machamp/experiment_one'
repo_path = path + '/MultiTask-Learning-for-Toxic-Language-Classification'
results_path = path + '/E1_results'
config_path = repo_path + '/config/' + experiment
dataset_config = config_path + '_data_'
parameter_config = config_path + '_parameter_'

# if __name__ == '__main__':

for fold in range(folds_number):
    split_data(fold)
    change_parameter_seeds(parameter_config + 'config.json')
    for model in models:
        output_path = experiment + '/' + model 
        train(
            dataset_config + model + '_config.json',
            device,
            output_path,
            parameter_config + 'config.json')