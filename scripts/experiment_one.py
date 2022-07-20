# dependencies
import argparse
from utils import MtlClass
import os

parser = argparse.ArgumentParser()
parser.add_argument("--information_config", default="", type=str, help="Modes configuration file")
parser.add_argument("--debug", default=False, help="Must be True or False", action='store_true')
args = parser.parse_args()

# check information config
if args.information_config == '':
    print('Specifying --information_config path is required')
    exit(1)

if not os.path.exists('../config' + '/' + args.information_config):
    print('The --information_config does not exist path')
    print('Enter with a valide path')
    exit(1)
    
    
#creating the mtl object
MTL = MtlClass(info_path=args.information_config, fetch_data=False, debug=args.debug)

# Save data folds and train the models
for idxs in zip(*[MTL.tasks[task_]['kfold'] for task_ in MTL.tasks.keys()]):
    #save
    for idx,task in zip(idxs, MTL.tasks.keys()):
        MTL.tasks[task]['df'].iloc[idx[0]].reset_index(drop=True).to_csv(MTL.data_path + '/' + MTL.tasks[task]['merged'].split('_')[0] + '_processed' + '_[TRAIN]' + '.tsv', header=None, index=False, sep="\t")
        MTL.tasks[task]['df'].iloc[idx[1]].reset_index(drop=True).to_csv(MTL.data_path + '/' + MTL.tasks[task]['merged'].split('_')[0] + '_processed' + '_[VAL]' + '.tsv', header=None, index=False, sep="\t")
    
    #train
    for model in ['mtl'] + ['stl_' + t.lower() for t in MTL.tasks.keys()]:
        MTL.train(model)
        
        #DEBUG add "Break"
        if args.debug == True:
            break
    
    #DEBUG add "Break"
    if args.debug == True:
        break

# average resuls
MTL.average()
# save resuls in drive
MTL.upload_data()
