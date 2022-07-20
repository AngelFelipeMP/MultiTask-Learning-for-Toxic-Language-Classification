import math
import os 
import pandas as pd

class Analyses:
    '''Class for train a mtl model'''
    def __init__(self, results):
        self.results = results
        self.path = '/' + '/'.join(os.getcwd().split('/')[1:-2])
        self.repo_path = '/' + '/'.join(os.getcwd().split('/')[1:-1])
        self.data_path = self.path + '/data'

    def ttest(self, v,n):
        z = 1.96
        SE = math.sqrt((v*(1-v))/n)
        inf = round(v - 1.96*SE,4)
        sup = round(v + 1.96*SE,4)
        print('Interval: +/- {}'.format(round(1.96*SE,4)))
        print('Value {} is contained in ({},{})'.format(v,inf,sup))
    
    def data(self, term):
        val_data = [file for file  in os.listdir(self.data_path) if 'VAL' in file and term in file.lower()]
        merde_data = [file for file  in os.listdir(self.data_path) if 'merge' in file and term in file.lower()]
        return val_data[0], merde_data[0]
            
    def data_size(self):
        for dataset in self.merde_data:
            df = pd.read_csv(self.data_path + '/' + dataset, sep="\t")
            n = df.shape[0]
            
            print('Dataset: {}'.format(dataset))
            print('Instances: {}'.format(n))
            
    def conf_interval(self, v, dataset):
        dataset_val, dataset_merge = self.data(dataset)
        # df_val = pd.read_csv(self.data_path + '/' + dataset_val, sep="\t")
        # n_val = df_val.shape[0]
        
        df_merge = pd.read_csv(self.data_path + '/' + dataset_merge, sep="\t")
        n_merge = df_merge.shape[0]
        
        print('Dataset: {}'.format(dataset_merge))
        print('Instances: '.format(n_merge))
        self.ttest(v,n_merge)
            
if __name__ == '__main__':
    
    
    # results = {'STL': {'exist':0.623809523809523, 'detoxis':0.615112310647964, 'hateval':0.810670810937881},
    # 'MTL':{'exist-detoxis':{'exist':0.683333333333333, 'detoxis':0.621767920255661},
    # 'exist-hateval':{'exist':0.652380952380952, 'hateval':0.814742237329483},
    # 'detoxis-hateval':{'detoxis':0.63452700972557, 'hateval':0.808410984277725}}}
    
    results = {'STL': {'exist':0.790740160383445, 'detoxis':0.615112310647964, 'hateval':0.810670810937881},
    'MTL':{'exist-detoxis':{'exist':0.795648139613482, 'detoxis':0.632434904575347},
    'exist-hateval':{'exist':0.798632131993732, 'hateval':0.820555013418197},
    'detoxis-hateval':{'detoxis':0.63452700972557, 'hateval':0.808410984277725},
    'detoxis-hateval-exist':{'detoxis':0.635180324316024, 'hateval':0.813288688659668, 'exist':0.799510553968107}}}
    
    
    Analyses = Analyses(results)
    for model_type in Analyses.results.keys():
        print('####################### {} ######################'.format(model_type))
        
        if model_type == 'STL':
            for model, performace in Analyses.results[model_type].items():
                print('\nModel: {}'.format(model))             
                # print('Confidence interval: {}'.format(Analyses.conf_interval(performace)))
                Analyses.conf_interval(performace, model)
                print('\n')
    
        else:
            for model in Analyses.results[model_type].keys():
                print('Model: {}'.format(model))
                for head, performace in Analyses.results[model_type][model].items():
                    print('Head: {}'.format(head))
                    # print('Performance: {}'.format(performace))
                    # print('Confidence interval: {}'.format(Analyses.conf_interval(performace)))
                    Analyses.conf_interval(performace, head)
                    print('\n')