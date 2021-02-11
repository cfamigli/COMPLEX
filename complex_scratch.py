
import os
import numpy as np
from pandas import read_pickle
import glob
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

def plot_heatmap_x_dimensionality_y_update(df, metric='hist_int', ystr='forecast', ylim=[0,1], subset='', var='NEE', xvar=''):
    xstr = 'dimensionality' if len(xvar)==0 else xvar
    x = df[xstr].values

    if (ystr=='diff'):
        y = df[metric + '_forecast'].values - df[metric + '_calibration'].values
        avgs_y = df.groupby(xstr).median()[metric + '_forecast'].values - df.groupby(xstr).median()[metric + '_calibration'].values
    else:
        y = df[metric + '_' + ystr].values
        avgs_y = df.groupby(xstr).median()[metric + '_' + ystr].values
        avgs_y[df.groupby(xstr).count()[metric + '_' + ystr].values < 4] = float('nan')

    keep = (np.isfinite(x) & (np.isfinite(y)))
    xk = x[keep]
    yk = y[keep]
    plt.figure(figsize=(4,4))
    plt.hist2d(x[keep], y[keep], bins=(20,40), cmap=LinearSegmentedColormap.from_list("", ["white","lightgray","dimgray"]), alpha=0.9, cmin=1)
    #plt.hexbin(x[keep], y[keep], mincnt=1, cmap=plt.cm.Greys, gridsize=(25,20), alpha=0.5, edgecolors='')

    avgs_x = np.unique(x)

    plt.scatter(avgs_x, avgs_y, facecolor='cornflowerblue', edgecolor='black', linewidth=1.5, marker='o', s=60, zorder=5)
    plt.ylim(ylim)
    plt.xlim([1,41])

    if metric=='hist_int':
        if ystr=='forecast':
            plt.ylabel('Histogram intersection \n(Average forecast)')
        else:
            plt.ylabel('Histogram\nintersection (%s)' % ystr)
    elif metric=='RMSE':
        plt.ylabel('RMSE (%s)' % ystr)
    else:
        plt.ylabel(metric + '_' + ystr)

    pred_y_best_fit = float('nan')
    rmse = [float('nan'), float('nan'), float('nan')]
    try:
        xfit = avgs_x
        yfit = avgs_y
        inds = (~np.isnan(xfit)) & (~np.isnan(yfit))

        degree = [0, 1, 4]
        pred_y = [[],[],[]]
        for i in range(len(degree)):
            if degree[i]==0:
                coef = np.array([0, np.nanmean(yfit[inds])])
            else:
                coef = np.polyfit(xfit[inds], yfit[inds], degree[i])
            poly1d_fn = np.poly1d(coef)
            pred_y[i] = poly1d_fn(xfit[inds])
            rmse[i] = np.sqrt(mean_squared_error(yfit[inds], pred_y[i]))
        pred_y_best_fit = pred_y[np.argmin(rmse)]
        #plt.plot(xfit[inds], pred_y_best_fit, linewidth=1.5, c='k', zorder=1)
    except Exception as e:
        print(e)

    plt.xlabel('Dimensionality') if len(xvar)==0 else plt.xlabel(xvar)
    plt.tight_layout()
    plt.savefig('../../plots/heatmaps/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_heatmap.pdf') if len(xvar)==0 else plt.savefig('../../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_' + xvar + '.pdf')
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
        avgs_y[df.groupby(xstr).count()[metric + '_' + ystr].values < 4] = float('nan')
    plt.figure(figsize=(3.15,3))
    plt.scatter(x, y, color='gainsboro', marker='.', alpha=0.2, zorder=0)
    avgs_x = np.unique(x)
    plt.scatter(avgs_x, avgs_y, facecolor='cornflowerblue', edgecolor='black', linewidth=1.5, marker='o', s=45, zorder=5)
    if (len(subset)==2) | (len(subset)==5): # model only
        plt.axvline(np.nanmean(x), c='silver', linewidth=0.5, zorder=0)
        plt.axhline(np.nanmean(y), c='silver', linewidth=0.5, zorder=0.5)
    #plt.ylim([ylim[0], np.nanmax(y)])
    plt.ylim(ylim)
    if metric=='hist_int':
        if ystr=='forecast':
            plt.ylabel('Histogram intersection \n(Average forecast)')
        else:
            plt.ylabel('Histogram\nintersection (%s)' % ystr)
    elif metric=='RMSE':
        plt.ylabel('RMSE (%s)' % ystr)
    else:
        plt.ylabel(metric + '_' + ystr)

    pred_y_best_fit = float('nan')
    rmse = [float('nan'), float('nan'), float('nan')]
    try:
        xfit = avgs_x
        yfit = avgs_y
        inds = (~np.isnan(xfit)) & (~np.isnan(yfit))

        degree = [0, 1, 1]
        pred_y = [[],[],[]]
        for i in range(len(degree)):
            if degree[i]==0:
                coef = np.array([0, np.nanmean(yfit[inds])])
            else:
                coef = np.polyfit(xfit[inds], yfit[inds], degree[i])
            poly1d_fn = np.poly1d(coef)
            pred_y[i] = poly1d_fn(xfit[inds])
            #rmse[i] = np.sqrt(mean_squared_error(yfit[inds], pred_y[i]))
        pred_y_best_fit = pred_y[np.argmin(rmse)]
        plt.plot(xfit[inds], pred_y_best_fit, linewidth=1.5, c='k', zorder=1)
    except Exception as e:
        print(e)

    plt.xlabel('Dimensionality') if len(xvar)==0 else plt.xlabel(xvar)
    plt.tight_layout()
    plt.savefig('../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_wfit.pdf') if len(xvar)==0 else plt.savefig('../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_' + xvar + '.pdf')
    plt.close()
    return xfit[inds], pred_y_best_fit, rmse

def plot_trend(df, metric='hist_int', ystr='forecast', ylim=[0,1], subset='', var='NEE', xvar='', col='cornflowerblue'):
    xstr = 'dimensionality' if len(xvar)==0 else xvar
    x = df[xstr].values

    if (ystr=='diff'):
        y = df[metric + '_forecast'].values - df[metric + '_calibration'].values
        avgs_y = df.groupby(xstr).median()[metric + '_forecast'].values - df.groupby(xstr).median()[metric + '_calibration'].values
    else:
        y = df[metric + '_' + ystr].values
        avgs_y = df.groupby(xstr).median()[metric + '_' + ystr].values
        avgs_y[df.groupby(xstr).count()[metric + '_' + ystr].values < 4] = float('nan')
    #plt.figure(figsize=(3.15,3))
    #plt.scatter(x, y, color='gainsboro', marker='.', alpha=0.2, zorder=0)
    avgs_x = np.unique(x)
    plt.scatter(avgs_x, avgs_y, facecolor=col, edgecolor=col, linewidth=1.5, marker='o', s=60, zorder=0, alpha=0.75)
    plt.ylim(ylim)
    if metric=='hist_int':
        if ystr=='forecast':
            plt.ylabel('Histogram intersection \n(Average forecast)')
        else:
            plt.ylabel('Histogram\nintersection (%s)' % ystr)
    elif metric=='RMSE':
        plt.ylabel('RMSE (%s)' % ystr)
    else:
        plt.ylabel(metric + '_' + ystr)

    pred_y_best_fit = float('nan')
    rmse = [float('nan'), float('nan'), float('nan')]
    try:
        xfit = avgs_x
        yfit = avgs_y
        inds = (~np.isnan(xfit)) & (~np.isnan(yfit))

        degree = [1]
        pred_y = [[],[],[]]
        for i in range(len(degree)):
            if degree[i]==0:
                coef = np.array([0, np.nanmean(yfit[inds])])
            else:
                coef = np.polyfit(xfit[inds], yfit[inds], degree[i])
            poly1d_fn = np.poly1d(coef)
            pred_y[i] = poly1d_fn(xfit[inds])
            #rmse[i] = np.sqrt(mean_squared_error(yfit[inds], pred_y[i]))
        pred_y_best_fit = pred_y[np.argmin(rmse)]
        plt.plot(xfit[inds], pred_y_best_fit, linewidth=2.5, c=col, zorder=1)
    except Exception as e:
        print(e)

    plt.xlabel('Dimensionality') if len(xvar)==0 else plt.xlabel('Number of parameters')
    plt.tight_layout()
    #plt.savefig('../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_wfit.pdf') if len(xvar)==0 else plt.savefig('../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_' + xvar + '_scratch.pdf')
    #plt.close()
    return avgs_x, avgs_y

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

def plot_time_series_with_spread(obs_data, pred_data, obs_unc, cal_period_stop, pred_data_prior=None, xlim=[None,None], ylim=[None,None], var='NEE', title=None, saveloc=''):
    o = np.copy(obs_data)
    p = np.copy(pred_data)
    o[obs_data==-9999] = float('nan')
    p[pred_data==-9999] = float('nan')
    if np.ndim(obs_unc)>0:
        obs_unc[obs_unc==-9999]=float('nan')
    plt.figure(figsize=(8,3))
    plt.axvspan(cal_period_stop, p.shape[1], alpha=0.4, color='lightgray',
        label='forecast window', zorder=0)
    plt.fill_between(np.arange(p.shape[1]), np.nanpercentile(p, 95, axis=0),
        np.nanpercentile(p, 5, axis=0), facecolor='lightsteelblue', alpha=0.8, label='ensemble spread (predicted)', zorder=1.5)
    plt.plot(np.nanmedian(p, axis=0), c='darkblue', linewidth=1.5,
        label='ensemble median (predicted)', zorder=2)
    plt.plot(o, c='crimson', linewidth=2, label='observed', zorder=4)
    plt.fill_between(np.arange(p.shape[1]), o+obs_unc, o-obs_unc,
        facecolor='lightpink', alpha=0.8, label='observational uncertainty', zorder=2)
    '''if pred_data_prior is not None:
        pp = np.copy(pred_data_prior)
        pp[pred_data_prior==-9999] = float('nan')
        plt.plot(np.nanmedian(pp, axis=0), c='darkblue', linewidth=1.5, label='prior', zorder=2)
        plt.fill_between(np.arange(pp.shape[1]), np.nanpercentile(pp, 95, axis=0),
        np.nanpercentile(pp, 5, axis=0), facecolor='lightsteelblue', alpha=0.8, label='ensemble spread (prior)', zorder=1)'''
    plt.ylabel(var)
    plt.xlabel('Months after Jan 1998')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    #plt.legend(loc='best')
    plt.savefig(saveloc + title + '.pdf')
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

    site_obs_dir = '../data/COMPLEX_input_files/FR-LBr'
    os.chdir(site_obs_dir)
    site_obs, site_obs_unc = computil.get_observations(var='NEE')
    site_calibration_indices = computil.calibration_period()

    for model, EDC, experiment, ylim in ['S2', 'EDC', 'exp2c', [-7,5]], ['C2', 'EDC', 'exp3a', [-7,5]], ['G4', 'noEDC', 'exp1f', [-17,9]]:
        print(model, EDC, experiment)
        pred_file = glob.glob('../../COMPLEX_v1.5/' + model + '/' + 'FR-LBr_' + EDC + '/*' + experiment + '*NEE*.csv')[0]
        print(pred_file)
        pred_data = computil.csv_to_np(pred_file, header=None)
        pred_prior_file = glob.glob('../../COMPLEX_v1.5/' + model + '/' + 'FR-LBr_noEDC' + '/*' + 'exp1f' + '*NEE*.csv')[0]
        pred_prior_data = computil.csv_to_np(pred_prior_file, header=None)

        cal_period_stop = 60
        observational_error = np.nanmean(site_obs_unc) if len(np.unique(site_obs_unc))==1 else site_obs_unc
        print(observational_error)

        plot_time_series_with_spread(site_obs, pred_data,
            observational_error, cal_period_stop, pred_prior_data, xlim=[0,131], ylim=ylim, var='NEE (gC m$^{-2}$ day$^{-1}$)',
            title=model + '_FR-LBr_' + EDC + '_' + experiment + '_' + 'NEE', saveloc='../../../misc/')

    '''data = read_pickle('../data/analysis_outputs/v1.5_NEE_082820.pkl')
    data = data.drop(data[data['dimensionality']==0].index)

    var = 'NEE'

    xstr = 'calibration'
    ystr = 'forecast'

    model_list = ['C1','C2','C3','C4','C6','C8','E1','G1','G2','G3','G4','S1','S2','S4']

    print('running for full dataset')
    #plot_scatter_x_dimensionality_y(data, ylim=[0,0.8], var=var)
    plt.figure(figsize=(3.5,3.5))
    #plot_trend(data.loc[(data.index.str.endswith('2b')) & (data.index.str.contains('_noEDC_'))], ylim=[1.4,3], var=var, metric='RMSE', subset='n_pars_obs', xvar='n_parameters', col='cornflowerblue')
    #plot_trend(data.loc[(data.index.str.endswith('f')) & (data.index.str.contains('_noEDC_'))], ylim=[1.4,3], var=var, metric='RMSE', subset='n_pars_obs', xvar='n_parameters', col='orangered')
    plot_trend(data.loc[(data.index.str.endswith('2b')) & (data.index.str.contains('_noEDC_'))], ylim=[1.4,3], var=var, metric='RMSE', subset='n_pars_obs', xvar='n_parameters', col='cornflowerblue')
    plot_trend(data.loc[(data.index.str.endswith('f'))], ylim=[1.4,3], var=var, metric='RMSE', subset='n_pars_obs', xvar='n_parameters', col='orangered')
    plt.show()

    data_obs_subset = data.loc[~(data.index.str.endswith('f'))]
    print('running for all obs only')
    #plot_scatter_x_dimensionality_y(data_obs_subset, subset='obs_only', ylim=[0,0.8], var=var)'''

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
