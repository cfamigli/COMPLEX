
import os
import sys
import complex_utilities as computil
from pandas import read_csv, DataFrame
import numpy as np
import glob
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

def plot_corr(corr, model, type='', dummy=0, dimensionality=''):
    plt.figure(figsize=(7,7))
    cmap = plt.cm.get_cmap('bwr_r')
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=np.bool)), cmap=cmap, vmin=-1, vmax=1, center=0,
        square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(model + '_' + type + '\nnumber of dummy pars = ' + str(dummy) + '\ndimensionality = ' + str(dimensionality))
    plt.tight_layout()
    plt.savefig('../../../../plots/heatmaps/dimensionality/dummy/' + model + '_' + type + '_' + str(dummy) + '.pdf')
    plt.close()
    return

def main():
    os.chdir('../data/COMPLEX_v1.4/')
    for model in computil.raw_complexity().sort_values('npars')['models']:
        os.chdir(model + '/US-Ha1_EDC/')

        exp = '1c'
        posterior = computil.csv_to_np(glob.glob('*exp' + exp + 'EDC*parameters_*.csv')[0], header=None)
        posterior_dimensionality = computil.do_PCA(posterior, posterior.shape[1])
        plot_corr(read_csv(glob.glob('*exp' + exp + 'EDC*parameters_*.csv')[0], header=None).corr(), model, type='posterior_exp' + exp,
            dummy=0, dimensionality=posterior_dimensionality)

        exp = '1f'
        prior = computil.csv_to_np(glob.glob('*exp' + exp + 'EDC*parameters_*.csv')[0], header=None)
        prior_dimensionality = computil.do_PCA(prior, prior.shape[1])
        plot_corr(DataFrame.from_records(prior).corr(), model, type='prior', dummy=0, dimensionality=prior_dimensionality)

        for i in range(1,11):
            prior = np.append(prior, prior[:,i].reshape((-1,1))*2, 1)#np.random.uniform(size=prior.shape[0]).reshape((-1,1)), 1)
            prior_dimensionality = computil.do_PCA(prior, prior.shape[1])
            plot_corr(DataFrame.from_records(prior).corr(), model, type='prior_dummy_fxn', dummy=i, dimensionality=prior_dimensionality)

            posterior = np.append(posterior, posterior[:,i].reshape((-1,1))*2, 1) #np.random.uniform(size=posterior.shape[0]).reshape((-1,1)), 1)
            posterior_dimensionality = computil.do_PCA(posterior, posterior.shape[1])
            plot_corr(DataFrame.from_records(posterior).corr(), model, type='posterior_dummy_fxn', dummy=i, dimensionality=posterior_dimensionality)

        os.chdir('../../')

    return

if __name__=='__main__':
    main()
