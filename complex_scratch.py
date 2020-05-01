
import os
import numpy as np
from pandas import read_pickle
import complex_utilities as computil
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def plot_heatmap_x_dimensionality_y(df, ystr='hist_int_forecast', ylim=[0,1], subset=''):
    x = df['dimensionality'].values
    y = df[ystr].values
    plt.figure(figsize=(5,5))
    plt.hist2d(x, y, bins=(25), cmap=plt.cm.Greys, cmin=1)
    plt.ylim([ylim[0], np.nanmax(y)])
    plt.ylabel(ystr)
    plt.xlabel('dimensionality')
    plt.savefig('../../plots/heatmaps/dimensionality/' + ystr + '_' + subset + '.pdf')
    plt.close()
    return

def plot_heatmap_x_performance_y(df, xstr='calibration', ystr='forecast',
    metric='hist_int', subset=''):
    x = df[metric + '_' + xstr].values
    if (ystr=='diff'):
        y = df[metric + '_forecast'].values - df[metric + '_calibration'].values
    else:
        y = df[metric + '_' + ystr].values
    plt.figure(figsize=(5,5))
    plt.hist2d(x, y, bins=(25,25), cmap=plt.cm.Greys, cmin=1)
    if (ystr!='diff'):
        plt.plot((0,1), c='k', linewidth=0.5)
    plt.ylabel(metric + '_' + ystr)
    plt.xlabel(metric + '_' + xstr)
    plt.savefig('../../plots/heatmaps/performance/' + metric + '_' + xstr + '_' + metric + '_' + ystr + '_' + subset + '.pdf')
    plt.close()
    return

def run_plots(df, subset_str=''):
    #plot_scatter_x_performance_y(df, xstr='calibration', ystr='forecast', subset=subset_str)
    #plot_scatter_x_performance_y(df, xstr='forecast', ystr='diff', subset=subset_str)
    #plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='diff', ylim=[-0.3,0.5], subset=subset_str)
    plot_scatter_x_dimensionality_y(df, metric='hist_int', subset=subset_str)
    return

def main():
    v1_2 = read_pickle('../data/analysis_outputs/v1.2_LAI_040320.pkl')

    var = 'LAI'

    xstr = 'calibration'
    ystr = 'forecast'

    os.chdir('scratch')

    model_list = ['C1','C2','C3','C4','C6','C8','E1','G1','G2','G3','G4','S1','S2','S4']

    v1_2_nee_subset = v1_2.loc[v1_2['nee']==1]
    print('running for nee only')
    computil.plot_scatter_x_dimensionality_y(v1_2_nee_subset, metric='RMSE', ystr='forecast', subset='nee_only', ylim=[0,6], var=var)

    v1_2_no_nee_subset = v1_2.loc[(v1_2['nee']==0) & ~(v1_2.index.str.endswith('f'))]
    print('running for other obs only')
    computil.plot_scatter_x_dimensionality_y(v1_2_no_nee_subset, metric='RMSE', ystr='forecast', subset='obs_no_nee_only', ylim=[0,6], var=var)

    ylim = [7,7,7,7,7,50]
    count = 0
    for experiment_letter in ['a','b','c','d','e','f']:
        to_plot = v1_2[v1_2.index.str.endswith(experiment_letter)]
        print(to_plot)
        computil.plot_scatter_x_dimensionality_y(to_plot, metric='RMSE', ystr='forecast', subset='exp' + experiment_letter, ylim=[0,ylim[count]], var=var)
        count += 1

    v1_2 = read_pickle('../../data/analysis_outputs/v1.2_NEE_040320.pkl')

    var = 'NEE'

    v1_2_nee_subset = v1_2.loc[v1_2['nee']==1]
    print('running for nee only')
    computil.plot_scatter_x_dimensionality_y(v1_2_nee_subset, metric='RMSE', ystr='forecast', subset='nee_only', ylim=[0,10], var=var)

    v1_2_no_nee_subset = v1_2.loc[(v1_2['nee']==0) & ~(v1_2.index.str.endswith('f'))]
    print('running for other obs only')
    computil.plot_scatter_x_dimensionality_y(v1_2_no_nee_subset, metric='RMSE', ystr='forecast', subset='obs_no_nee_only', ylim=[0,10], var=var)

    ylim = [5, 5, 5, 10, 5, 7]
    count = 0
    for experiment_letter in ['a','b','c','d','e','f']:
        to_plot = v1_2[v1_2.index.str.endswith(experiment_letter)]
        print(to_plot)
        computil.plot_scatter_x_dimensionality_y(to_plot, metric='RMSE', ystr='forecast', subset='exp' + experiment_letter, ylim=[0,ylim[count]], var=var)
        count += 1


    '''computil.plot_scatter_x_performance_y_dimensionality(v1_2,
        ['_EDC', '_noEDC'], xstr=xstr, ystr=ystr, subset='EDCs_test')

    computil.plot_scatter_x_performance_y_dimensionality(v1_2,
        ['nee', 'no_nee'], xstr=xstr, ystr=ystr, subset='nee')

    computil.plot_scatter_x_performance_y_dimensionality(v1_2,
        model_list, xstr=xstr, ystr=ystr, subset='models')

    computil.plot_scatter_x_performance_y_dimensionality(v1_2,
        ['nee_EDC', 'no_nee_noEDC'], xstr=xstr, ystr=ystr, subset='nee_EDC')

    computil.plot_scatter_x_dimensionality_y_resampled(v1_2,metric='hist_int', ystr='forecast', subset='', ylim=[0,0.6])
    computil.plot_scatter_x_dimensionality_y_resampled(v1_2,metric='hist_int', ystr='diff', subset='', ylim=[-0.3,0.3])'''

    return

if __name__=='__main__':
    main()
