from utils import split_data, change_parameter_seeds

foldes_number = 10
experiment = 'E1'
models = ['mtl', 'stl_exist', 'stl_detoxis']
device = 0  #gpu 0 / cpu 1
data_path = '/data'
path = '/content/drive/MyDrive/Code/MTL_2021/machamp/experiment_one'
repo_path = path + '/MultiTask-Learning-for-Toxic-Language-Classification'
results_path = path + '/E1_results'
dataset_config = repo_path + '/config/' + experiment + '_'

for fold in range(foldes_number):
    split_data(fold)
    for model in models:
        change_parameter_seeds( dataset_config + model + '_config.json')
        output_path = experiment + '/' + model
    
python3 train.py --dataset_config /content/MultiTask-Learning-for-Toxic-Language-Classification/config/EXIST_DETOXIS_config.json 
--device 0 
--name E0 
--parameters_config /content/MultiTask-Learning-for-Toxic-Language-Classification/config/E0_params_config.json

###### How to train a python script insed another python script ? #########