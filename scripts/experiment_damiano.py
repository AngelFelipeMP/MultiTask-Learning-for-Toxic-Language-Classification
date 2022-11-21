# dependencies
import argparse
from utils_two import MtlClass
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
    print('Enter with a valid path')
    exit(1)
    
    
#creating the mtl object
MTL = MtlClass(info_path=args.information_config, fetch_data=False, debug=args.debug)

#train
for model in MTL.models():
    print(model)
    MTL.train(model)
    
    #DEBUG add "Break"
    if args.debug == True:
        break