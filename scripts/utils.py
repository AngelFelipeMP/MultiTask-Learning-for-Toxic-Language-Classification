import os

def split_data(args):
    #### construction
    return

def change_parameter_seeds(args):
    #### construction
    return

def train(dataset_config=str(), device=int(), output_path=str(), parameter_config=str()):
    code_line = 'python3 train.py --dataset_config ' + dataset_config
    code_line = code_line + ' --device ' + str(device)
    code_line = code_line + ' --name ' + output_path
    code_line = code_line + ' --parameters_config ' + parameter_config
    
    os.system(code_line)

