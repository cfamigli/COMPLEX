
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

    models = computil.raw_complexity().sort_values('npars')['models']
    for model in models:
        computil.plot_density(data, subset_main=model, subset_sub=['_EDC','_noEDC'], title='EDC', var=var)

        numeric_experiments = computil.get_experiments(type='numeric')
        #numeric_experiments.remove('exp1')
        #numeric_experiments.remove('exp3')
        computil.plot_density(data, subset_main=model, subset_sub=numeric_experiments, title='num_exp',  var=var)

        letter_experiments_full = ['a', 'b', 'c', 'd', 'e', 'f']
        computil.plot_density(data, subset_main=model, subset_sub=letter_experiments_full, title='let_exp',  var=var)

        letter_experiments_individual = ['a', 'd']
        computil.plot_density(data, subset_main=model, subset_sub=letter_experiments_individual, title='let_exp_individual',  var=var)

        nee_experiments = computil.get_experiments(type='nee')
        no_nee_experiments = computil.get_experiments(type='no_nee')
        computil.plot_density(data, subset_main=model, subset_sub='', title='nee',
            subset_sub_list=[nee_experiments, no_nee_experiments], var=var)

    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=['_EDC','_noEDC'], title='EDC', var=var)
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=['_EDC','_noEDC'], title='EDC', var=var, zero_point='prior')
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=['_EDC','_noEDC'], title='EDC', var=var, zero_point='prior', fractional=True)
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=numeric_experiments, title='num_exp', var=var)
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=numeric_experiments, title='num_exp', var=var, zero_point='prior')
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=numeric_experiments, title='num_exp', var=var, zero_point='prior', fractional=True)
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=letter_experiments_full, title='let_exp',  var=var)
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=letter_experiments_full, title='let_exp',  var=var, zero_point='prior')
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=letter_experiments_full, title='let_exp',  var=var, zero_point='prior', fractional=True)
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=letter_experiments_individual, title='let_exp_individual',  var=var)
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=letter_experiments_individual, title='let_exp_individual',  var=var, zero_point='prior')
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=letter_experiments_individual, title='let_exp_individual',  var=var, zero_point='prior', fractional=True)
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=['AU-How', 'FI-Hyy', 'FR-LBr', 'FR-Pue', 'GF-Guy', 'US-Ha1'], title='sites',  var=var)
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=['AU-How', 'FI-Hyy', 'FR-LBr', 'FR-Pue', 'GF-Guy', 'US-Ha1'], title='sites',  var=var, zero_point='prior')
    computil.plot_dimensionality_medians(data, subset_main=models, subset_sub=['AU-How', 'FI-Hyy', 'FR-LBr', 'FR-Pue', 'GF-Guy', 'US-Ha1'], title='sites',  var=var, zero_point='prior', fractional=True)

    computil.plot_dimensionality_reduction_bar(data, subset_main=models, title='bar', type='constrainability')
    computil.plot_dimensionality_reduction_bar(data, subset_main=models, title='bar', type='dimensionality')
    computil.plot_dimensionality_reduction_bar(data, subset_main=models, title='bar', type='')
    return

if __name__=='__main__':
    main()
