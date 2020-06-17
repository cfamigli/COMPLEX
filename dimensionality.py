
import os
import sys
import complex_utilities as computil
from pandas import read_pickle
import numpy as np
import seaborn as sns

def main():
    os.chdir('../data/analysis_outputs')

    version = sys.argv[1]
    var = sys.argv[2]
    date = '052320'

    dataset_str = version + '_' + var + '_' + date + '.pkl'
    data = read_pickle(dataset_str)

    models = ['C1','C2','C3','C4','C5','C6','C7','C8','E1','G1','G2','G3','G4','S1','S2','S4']
    for model in models:
        computil.plot_density(data, subset_main=model, subset_sub=['_EDC','_noEDC'], title='EDC', var=var)

        numeric_experiments = computil.get_experiments(type='numeric')
        #numeric_experiments.remove('exp1')
        #numeric_experiments.remove('exp3')
        computil.plot_density(data, subset_main=model, subset_sub=numeric_experiments, title='num_exp',  var=var)

        letter_experiments = ['a', 'b', 'c', 'd', 'e', 'f']
        computil.plot_density(data, subset_main=model, subset_sub=letter_experiments, title='let_exp',  var=var)

        nee_experiments = computil.get_experiments(type='nee')
        no_nee_experiments = computil.get_experiments(type='no_nee')
        computil.plot_density(data, subset_main=model, subset_sub='', title='nee',
            subset_sub_list=[nee_experiments, no_nee_experiments], var=var)

    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=['_EDC','_noEDC'], title='EDC', var=var)
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=numeric_experiments, title='num_exp', var=var)
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=letter_experiments, title='let_exp',  var=var)
    return

if __name__=='__main__':
    main()
