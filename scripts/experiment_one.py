# dependencies
import argparse
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from utils import MtlClass

parser = argparse.ArgumentParser()
parser.add_argument("--information_config", default="", type=str, help="Modes configuration file")
parser.add_argument("--debug", default=False, help="Must be True or False", action='store_true')
args = parser.parse_args()

# check information config
if args.information_config == '':
    print('Specifying --information_config path is required')
    exit(1)
    
#creating the mtl object
MTL = MtlClass(info_path=args.information_config, fetch_data=True, debug=args.debug)

# Save data folds and train the models
split_sequence = MTL.tasks.keys()

for idxs in zip(*[MTL.tasks[data]['kfold'] for data in split_sequence]):
    #save
    for idx,task in zip(idxs, split_sequence):
        MTL.tasks[task]['df'].iloc[idx[0]].reset_index(drop=True).to_csv(MTL.data_path + '/' + MTL.tasks[task]['file'].split('_')[0] + '_processed' + '_[TRAIN]' + '.tsv', header=None, sep="\t")
        MTL.tasks[task]['df'].iloc[idx[1]].reset_index(drop=True).to_csv(MTL.data_path + '/' + MTL.tasks[task]['file'].split('_')[0] + '_processed' + '_[VAL]' + '.tsv', header=None, sep="\t")
    
    #train
    for model in ['mtl'] + ['stl_' + t.lower() for t in MTL.tasks.keys()]:
        MTL.train(model)
            
        #DEBUG add "Break"
        if args.debug == True:
            break
    
    #DEBUG add "Break"
    if args.debug == True:
        break

# #TODO add the avg func
# # average(info['folds_number'], repo_path + '/machamp/logs/' + info['experiment'], models)
