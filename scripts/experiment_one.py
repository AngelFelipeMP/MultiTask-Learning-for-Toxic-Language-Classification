
data_path = 'data'
path = '/content/drive/MyDrive/Code/MTL_2021/machamp/experiment_one'
repo_path = path + '/MultiTask-Learning-for-Toxic-Language-Classification'
results_path = path + '/E1_results'

#MTL
## train
dataset_config = repo_path + '/config/EXIST_DETOXIS_config.json'
device = 0  #gpu 0 / cpu 1
output_train = repo_path + 'E1/mtl'
## test
model_weights = repo_path + '/logs/E1/' ???????????? '/model.tar.gz'
dataset_test = '/content/data/EXIST2021_test_with_labeled_processed_[TEST].tsv'
head =
device = 0
