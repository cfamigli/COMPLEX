
import os
import complex_utilities as computil
from pandas import read_pickle
import numpy as np
import seaborn as sns

def main():
    os.chdir('../data/analysis_outputs')
    v1 = read_pickle('v1.4_NEE_052020.pkl')

    models = ['C1','C2','C3','C4','C5','C6','C7','C8','E1','G1','G2','G3','G4','S1','S2','S4']
    var = 'NEE'
    for model in models:
        computil.plot_density(v1, subset_main=model, subset_sub=['_EDC','_noEDC'], title='EDC', var=var)

        numeric_experiments = computil.get_experiments(type='numeric')
        numeric_experiments.remove('exp1')
        numeric_experiments.remove('exp3')
        computil.plot_density(v1, subset_main=model, subset_sub=numeric_experiments, title='exp',  var=var)

        nee_experiments = computil.get_experiments(type='nee')
        no_nee_experiments = computil.get_experiments(type='no_nee')
        computil.plot_density(v1, model, subset_sub='', title='nee',
            subset_sub_list=[nee_experiments, no_nee_experiments], var=var)

    return

if __name__=='__main__':
    main()
