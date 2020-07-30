
import os
import numpy as np
from pandas import read_pickle
import complex_utilities as computil
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

def plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='forecast', ylim=[0,1], subset='', var='NEE', xvar=''):
    xstr = 'dimensionality' if len(xvar)==0 else xvar
    x = df[xstr].values

    if (ystr=='diff'):
        y = df[metric + '_forecast'].values - df[metric + '_calibration'].values
        avgs_y = df.groupby(xstr).median()[metric + '_forecast'].values - df.groupby(xstr).median()[metric + '_calibration'].values
    else:
        y = df[metric + '_' + ystr].values
        avgs_y = df.groupby(xstr).median()[metric + '_' + ystr].values
        avgs_y[df.groupby(xstr).count()[metric + '_' + ystr].values < 3] = float('nan')
        y_25 = df.groupby(xstr).quantile(.25)[metric + '_' + ystr].values
        y_25[df.groupby(xstr).count()[metric + '_' + ystr].values < 3] = float('nan')
        y_75 = df.groupby(xstr).quantile(.75)[metric + '_' + ystr].values
        y_75[df.groupby(xstr).count()[metric + '_' + ystr].values < 3] = float('nan')

    plt.figure(figsize=(5,5))
    #plt.scatter(x, y, color='gainsboro', marker='.', alpha=0.2, zorder=0)

    avgs_x = np.unique(x)
    plt.fill_between(avgs_x, y_25, y_75, color='gainsboro', edgecolor='white', alpha=0.3)
    plt.scatter(avgs_x, avgs_y, facecolor='cornflowerblue', edgecolor='black', linewidth=1.5, marker='o', s=60)

    plt.ylim(ylim)
    if metric=='hist_int':
        plt.ylabel('histogram overlap (%s)' % ystr)
    elif metric=='RMSE':
        plt.ylabel('RMSE (%s)' % ystr)
    else:
        plt.ylabel(metric + '_' + ystr)

    try:
        inds = (~np.isnan(x)) & (~np.isnan(y))
        coef = np.polyfit(x[inds], y[inds], 4)
        poly1d_fn = np.poly1d(coef)
        plt.plot(avgs_x, poly1d_fn(avgs_x), linewidth=1.5, c='k', zorder=1)
    except:
        pass

    plt.xlabel('dimensionality') if len(xvar)==0 else plt.xlabel('prior minus posterior dimensionality')
    plt.tight_layout()
    plt.savefig('../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_scratch.pdf') if len(xvar)==0 else plt.savefig('../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_' + xvar + '.pdf')
    plt.close()
    return

def plot_scatter_x_dimensionality_y_colored_by_subset(df, metric='hist_int', ystr='forecast', ylim=[0,1], subset='', var='NEE', xvar=''):
    xstr = 'dimensionality' if len(xvar)==0 else xvar
    x = df[xstr].values

    if (ystr=='diff'):
        y = df[metric + '_forecast'].values - df[metric + '_calibration'].values
        avgs_y = df.groupby(xstr).median()[metric + '_forecast'].values - df.groupby(xstr).median()[metric + '_calibration'].values
    else:
        y = df[metric + '_' + ystr].values
        avgs_y = df.groupby(xstr).median()[metric + '_' + ystr].values
        avgs_y[df.groupby(xstr).count()[metric + '_' + ystr].values < 3] = float('nan')

    plt.figure(figsize=(5,5))
    color = np.ones(len(x))
    color[df.index.str.endswith('f')] = 0
    plt.scatter(x, y, c=color, cmap=LinearSegmentedColormap.from_list("", ["blue","gainsboro"]), marker='.', alpha=0.4, zorder=0)

    '''for endlet in ['a','b','c','d','e','f']:
        color[df.index.str.endswith(endlet)] += ['a','b','c','d','e','f'].index(endlet)
        print(color)

    plt.scatter(x, y, c=color, cmap=plt.cm.Accent, marker='.', alpha=0.4, zorder=0)'''

    avgs_x = np.unique(x)

    plt.scatter(avgs_x, avgs_y, facecolor='k', edgecolor='black', linewidth=1.5, marker='o', s=60)

    plt.ylim(ylim)
    if metric=='hist_int':
        plt.ylabel('histogram overlap (%s)' % ystr)
    elif metric=='RMSE':
        plt.ylabel('RMSE (%s)' % ystr)
    else:
        plt.ylabel(metric + '_' + ystr)

    try:
        inds = (~np.isnan(avgs_x)) & (~np.isnan(avgs_y))
        coef = np.polyfit(avgs_x[inds], avgs_y[inds], 4)
        poly1d_fn = np.poly1d(coef)
        plt.plot(avgs_x[inds], poly1d_fn(avgs_x[inds]), linewidth=1.5, c='k', zorder=1)
    except:
        pass

    plt.xlabel('dimensionality') if len(xvar)==0 else plt.xlabel('prior minus posterior dimensionality')
    plt.tight_layout()
    plt.savefig('../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_scratch_let.pdf') if len(xvar)==0 else plt.savefig('../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_' + xvar + '.pdf')
    plt.close()
    return

def compute_statistics_of_fit(avgs_x, pred_y):
    peak_location = avgs_x[np.argmax(pred_y)]
    range_y = pred_y[-1] - pred_y[0]
    return peak_location, range_y


def run_plots(df, subset_str=''):
    #plot_scatter_x_performance_y(df, xstr='calibration', ystr='forecast', subset=subset_str)
    #plot_scatter_x_performance_y(df, xstr='forecast', ystr='diff', subset=subset_str)
    #plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='diff', ylim=[-0.3,0.5], subset=subset_str)
    plot_scatter_x_dimensionality_y(df, metric='hist_int', subset=subset_str)
    return

def main():
    data = read_pickle('../data/analysis_outputs/v1.4_NEE_062520.pkl')
    data = data.drop(data[data['dimensionality']==0].index)

    var = 'NEE'

    xstr = 'calibration'
    ystr = 'forecast'

    model_list = ['C1','C2','C3','C4','C6','C8','E1','G1','G2','G3','G4','S1','S2','S4']

    print('running for full dataset')
    plot_scatter_x_dimensionality_y(data, ylim=[0,0.8], var=var)


    data_obs_subset = data.loc[~(data.index.str.endswith('f'))]
    print('running for all obs only')
    plot_scatter_x_dimensionality_y(data_obs_subset, subset='obs_only', ylim=[0,0.8], var=var)

    '''plt.figure(figsize=(5,5))
    for model in model_list:
        to_plot = computil.subset_df_by_substring(data, model)
        plot_scatter_x_dimensionality_y(data.loc[to_plot], metric='hist_int', ystr='forecast', subset=model, ylim=[0,0.8], var=var, xvar='prior_minus_post')
        plt.draw()
        plt.pause(0.1)
    plt.show()



    os.chdir('scratch')



    computil.plot_scatter_x_process_y_skill(data, process='n_parameters', ystr='forecast', metric='hist_int', var='NEE')'''

    '''data_nee_subset = data.loc[data['nee']==1]
    print('running for nee only')
    computil.plot_scatter_x_dimensionality_y(data_nee_subset, metric='RMSE', ystr='forecast', subset='nee_only', ylim=[0,6], var=var)

    data_no_nee_subset = data.loc[(data['nee']==0) & ~(data.index.str.endswith('f'))]
    print('running for other obs only')
    computil.plot_scatter_x_dimensionality_y(data_no_nee_subset, metric='RMSE', ystr='forecast', subset='obs_no_nee_only', ylim=[0,6], var=var)

    ylim = [7,7,7,7,7,50]
    count = 0
    for experiment_letter in ['a','b','c','d','e','f']:
        to_plot = data[data.index.str.endswith(experiment_letter)]
        print(to_plot)
        computil.plot_scatter_x_dimensionality_y(to_plot, metric='RMSE', ystr='forecast', subset='exp' + experiment_letter, ylim=[0,ylim[count]], var=var)
        count += 1

    data = read_pickle('../../data/analysis_outputs/v1.2_NEE_040320.pkl')

    var = 'NEE'

    data_nee_subset = data.loc[data['nee']==1]
    print('running for nee only')
    computil.plot_scatter_x_dimensionality_y(data_nee_subset, metric='RMSE', ystr='forecast', subset='nee_only', ylim=[0,10], var=var)

    data_no_nee_subset = data.loc[(data['nee']==0) & ~(data.index.str.endswith('f'))]
    print('running for other obs only')
    computil.plot_scatter_x_dimensionality_y(data_no_nee_subset, metric='RMSE', ystr='forecast', subset='obs_no_nee_only', ylim=[0,10], var=var)

    ylim = [5, 5, 5, 10, 5, 7]
    count = 0
    for experiment_letter in ['a','b','c','d','e','f']:
        to_plot = data[data.index.str.endswith(experiment_letter)]
        print(to_plot)
        computil.plot_scatter_x_dimensionality_y(to_plot, metric='RMSE', ystr='forecast', subset='exp' + experiment_letter, ylim=[0,ylim[count]], var=var)
        count += 1'''


    '''computil.plot_scatter_x_performance_y_dimensionality(data,
        ['_EDC', '_noEDC'], xstr=xstr, ystr=ystr, subset='EDCs_test')

    computil.plot_scatter_x_performance_y_dimensionality(data,
        ['nee', 'no_nee'], xstr=xstr, ystr=ystr, subset='nee')

    computil.plot_scatter_x_performance_y_dimensionality(data,
        model_list, xstr=xstr, ystr=ystr, subset='models')

    computil.plot_scatter_x_performance_y_dimensionality(data,
        ['nee_EDC', 'no_nee_noEDC'], xstr=xstr, ystr=ystr, subset='nee_EDC')

    computil.plot_scatter_x_dimensionality_y_resampled(data,metric='hist_int', ystr='forecast', subset='', ylim=[0,0.6])
    computil.plot_scatter_x_dimensionality_y_resampled(data,metric='hist_int', ystr='diff', subset='', ylim=[-0.3,0.3])'''

    return

if __name__=='__main__':
    main()
