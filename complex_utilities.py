
import numpy as np
from pandas import read_csv, to_numeric, DataFrame
import glob
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import gaussian_kde, pearsonr

def csv_to_np(file, header=0):
    # return a numpy array of floats
    return read_csv(file, header=header).apply(to_numeric, errors='coerce').values * 1.

def csv_to_df(file, header=0):
    # return a dataframe
    return read_csv(file, header=header)

def subset_df_by_substring(df, subset_str):
    subset = [row for row in df.index if subset_str in row]
    return subset

def subset_list_by_substring(l, subset_str):
    subset = [x for x in l if subset_str in x]
    return subset

def subset_list_by_list_of_substrings(l, subset_str_list):
    subset = []
    for el in subset_str_list:
        subset.extend([x for x in l if el in x])
    return subset

def site_years():
    # change values in this function manually if new sites are added
    # return a dataframe
    data = [['AU-How',14],['FI-Hyy',16],['FR-LBr',11],
        ['FR-Pue',15],['GF-Guy',15],['US-Ha1',15]]
    return DataFrame(data,columns=['sites','nyears'])

def raw_complexity():
    data = [['C1',24],['C2',33],['C3',35],['C4',37],['C5',36],['C6',24],['C7',28],['C8',36],['E1',18],
        ['G1',38],['G2',41],['G3',44],['G4',44],['S1',12],['S2',14],['S4',18]]
    return DataFrame(data,columns=['models','npars'])

def processes_discrete(model=None):
    data = [['C1',24.,np.nan,np.nan,np.nan,0.,0.,1.,1.,0.,0.,0.,2.,4.,0.,6.,0],
            ['C2',33.,np.nan,np.nan,np.nan,1.,1.,2.,2.,0.,0.,0.,2.,4.,1.,7.,1],
            ['C3',35.,np.nan,np.nan,np.nan,1.,1.,2.,2.,0.,0.,0.,2.,4.,1.,7.,1],
            ['C4',37.,np.nan,np.nan,np.nan,1.,1.,2.,2.,0.,0.,0.,2.,4.,1.,7.,1],
            ['C5',36.,np.nan,np.nan,np.nan,1.,1.,1.,2.,3.,2.,1.,np.nan,np.nan,np.nan,np.nan,np.nan],
            ['C6',24.,np.nan,np.nan,np.nan,0.,0.,1.,1.,0.,1.,2.,3.,4.,0.,np.nan,np.nan],
            ['C7',28.,np.nan,np.nan,np.nan,1.,0.,1.,1.,0.,1.,2.,3.,4.,5.,np.nan,np.nan],
            ['C8',36.,np.nan,np.nan,np.nan,1.,0.,2.,1.,0.,0.,2.,8.,1.,np.nan,np.nan,np.nan],
            ['E1',18.,np.nan,np.nan,np.nan,0.,0.,0.,0.,0.,0.,0.,3.,3.,0.,np.nan,np.nan],
            ['G1',38.,np.nan,np.nan,np.nan,0.,0.,2.,3.,1.,1.,2.,3.,4.,0.,np.nan,np.nan],
            ['G2',41.,np.nan,np.nan,np.nan,1.,0.,2.,3.,1.,1.,2.,3.,4.,3.,np.nan,np.nan],
            ['G3',44.,np.nan,np.nan,np.nan,0.,0.,2.,4.,2.,1.,2.,3.,4.,0.,np.nan,np.nan],
            ['G4',44.,np.nan,np.nan,np.nan,1.,0.,2.,4.,2.,1.,2.,3.,4.,3.,np.nan,np.nan],
            ['S1',12.,np.nan,np.nan,np.nan,0.,0.,0.,0.,0.,0.,0.,1.,2.,0.,np.nan,np.nan],
            ['S2',14.,np.nan,np.nan,np.nan,0.,0.,1.,1.,0.,0.,0.,3.,np.nan,np.nan,np.nan,np.nan],
            ['S4',18.,np.nan,np.nan,np.nan,0.,0.,1.,1.,0.,0.,0.,3.,2.,0.,np.nan,np.nan]]

    return_df = DataFrame(data,columns=['model','n_parameters','prior_dim','prior_minus_post','prior_minus_post_normalized',
            'PAW','Rh','labile_c_lifespan','phenology','CUE',
            'photosynthesis_module','stomatal_conductance','n_DOM_pool','n_live_C_pool',
            'n_water_pool','total_pool','water_stress_on_GPP'])

    if model is not None:
        return_df = return_df[return_df['model']==model]

    return return_df

def get_experiments(type='all'):
    if type=='numeric':
        experiments = []
        for num in range(1,5):
            experiments.append('exp' + str(num))
    elif type=='letters':
        experiments = []
        for num in range(1,5):
            for letter in ['a', 'b', 'c', 'd', 'e', 'f']:
                experiments.append('exp' + str(num) + letter)
    elif type=='nee':
        experiments = []
        for num in range(1,5):
            for letter in ['a', 'b', 'c']:
                experiments.append('exp' + str(num) + letter)
    elif type=='no_nee':
        experiments = []
        for num in range(1,5):
            for letter in ['d', 'e', 'f']:
                experiments.append('exp' + str(num) + letter)
    else:
        experiments = []
        for num in range(1,5):
            for letter in ['','a', 'b', 'c', 'd', 'e', 'f']:
                experiments.append('exp' + str(num) + letter)
    return experiments

def calibration_period():
    # must be in a site folder
    files = sorted(glob.glob('*timeseries_obs.csv'))

    cal_df = DataFrame()
    for file in files:
        experiment = file[7:12]
        obs_data = csv_to_np(file)[:,1:]
        obs_data[obs_data==-9999] = float('nan')
        cal_period_stop = cal_df.values.max() if np.all(np.isnan(obs_data)) else np.max(np.nonzero(np.sum(np.isfinite(obs_data), axis=1)))
        cal_df = DataFrame(data=[cal_period_stop+1], columns=[experiment]) if cal_df.empty else cal_df.join(DataFrame(data=[cal_period_stop+1], columns=[experiment]))

    # each column is the calibration window stop index for an experiment
    return cal_df

def get_observations(var='NEE'):
    # must be in a site folder
    data = csv_to_df(glob.glob('master/*timeseries_obs.csv')[0])
    cols = [col for col in data.columns if var in col]
    obs = data[cols[0]]
    obs_unc = data[cols[1]]
    assert var+'_unc' in data[cols[1]].name
    return([np.asarray(obs), np.asarray(obs_unc)])

def do_PCA(X, n_components):
    # perform PCA on matrix X (rows=num_ensembles x cols=num_parameters=n_components).
    # returns number of principal components needed for >95% variance explained.
    # (alternatively can edit to return the components themselves, etc)

    if (X==X[0]).all():
        reduced_dimensionality = float('nan')
    else:
        X_scaled = StandardScaler().fit_transform(X) # standardize features by removing the mean and scaling to unit variance
        X_scaled = X_scaled[~np.isnan(X_scaled).any(axis=1)] # remove any row that has NaN
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        reduced_dimensionality = np.argwhere(np.cumsum(pca.explained_variance_ratio_)>0.95)[0][0]
    return reduced_dimensionality

def calc_R2_RMSE(obs_data, pred_data):
    inds = (obs_data == -9999) | (pred_data == -9999) | np.isnan(obs_data) | np.isnan(pred_data)
    try:
        r, _ = pearsonr(obs_data[~inds], pred_data[~inds])
        #R2 = r2_score(obs_data[~inds], pred_data[~inds])
        RMSE = np.sqrt(mean_squared_error(obs_data[~inds], pred_data[~inds])) / abs(np.nanstd(obs_data[~inds]))
    except:
        r = float('nan')
        RMSE = float('nan')
    return (r, RMSE)

def append_to_df(df, model, site_EDC, experiment, good_flag, dimensionality,
    hist_int_calibration, hist_int_forecast, R2_calibration, R2_forecast,
    RMSE_calibration, RMSE_forecast, nee_bool):
    processes = processes_discrete(model=model)

    # returns dataframe that will comtain all COMPLEX trials (columns) and metrics (rows)
    df = DataFrame(data=np.append([good_flag, dimensionality, hist_int_calibration, hist_int_forecast,
        R2_calibration, R2_forecast, RMSE_calibration, RMSE_forecast, nee_bool], processes.iloc[0,1:].values).astype(np.float),
        columns=[model + '_' + site_EDC + '_' + experiment],
        index=['good_flag', 'dimensionality', 'hist_int_calibration', 'hist_int_forecast',
        'R2_calibration', 'R2_forecast', 'RMSE_calibration', 'RMSE_forecast', 'nee'] + list(processes)[1:]) if df.empty else df.join(
        DataFrame(data=np.append([good_flag, dimensionality, hist_int_calibration, hist_int_forecast,
        R2_calibration, R2_forecast, RMSE_calibration, RMSE_forecast, nee_bool], processes.iloc[0,1:].values).astype(np.float),
        columns=[model + '_' + site_EDC + '_' + experiment],
        index=['good_flag', 'dimensionality', 'hist_int_calibration', 'hist_int_forecast',
        'R2_calibration', 'R2_forecast', 'RMSE_calibration', 'RMSE_forecast', 'nee'] + list(processes)[1:]))
    return df

def plot_time_series(obs_data, pred_data, model=None):
    o = np.copy(obs_data)
    p = np.copy(pred_data)
    o[obs_data==-9999] = float('nan')
    p[pred_data==-9999] = float('nan')
    plt.figure(figsize=(9,4))
    plt.axvspan(len(o)/2, len(o), alpha=0.4, color='lightgray', label='forecast', zorder=0)
    plt.plot(range(len(o)), o, 'crimson', linewidth=2, label='obs', zorder=2)
    plt.plot(range(len(p)), p, 'darkblue', linewidth=2.5, label='ensemble mean', zorder=3)
    plt.xlabel('months since start')
    plt.ylabel('nee')
    plt.xlim([0,len(o)])
    plt.title(model)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
    plt.close()
    return

def plot_time_series_with_spread(obs_data, pred_data, obs_unc, cal_period_stop, xlim=[None,None], var='NEE', title=None, saveloc='../../../../plots/time_series_with_spread/'):
    o = np.copy(obs_data)
    p = np.copy(pred_data)
    o[obs_data==-9999] = float('nan')
    p[pred_data==-9999] = float('nan')
    if np.ndim(obs_unc)>0:
        obs_unc[obs_unc==-9999]=float('nan')
    plt.figure(figsize=(11,4))
    plt.axvspan(cal_period_stop, p.shape[1], alpha=0.4, color='lightgray',
        label='forecast window', zorder=0)
    plt.fill_between(np.arange(p.shape[1]), np.nanpercentile(p, 95, axis=0),
        np.nanpercentile(p, 5, axis=0), facecolor='lightsteelblue', alpha=0.8, label='ensemble spread (predicted)', zorder=1)
    plt.plot(np.nanmedian(p, axis=0), c='darkblue', linewidth=2.5,
        label='ensemble median (predicted)', zorder=2)
    plt.plot(o, c='crimson', linewidth=2, label='observed', zorder=4)
    plt.fill_between(np.arange(p.shape[1]), o+obs_unc, o-obs_unc,
        facecolor='lightpink', alpha=0.8, label='observational uncertainty', zorder=1.5)
    plt.ylabel(var)
    plt.xlabel('months after start')
    plt.xlim(xlim)
    plt.tight_layout()
    #plt.legend(loc='best')
    plt.savefig(saveloc + title + '.pdf')
    plt.close()
    return

def running_mean(vec, N):
    rm = np.zeros(len(vec))
    for i in range(len(vec)):
        start = int(i-(N-1)/2)
        stop = int(i+(N-1)/2)
        print(start)
        print(stop)
        if start<0:
            rm[i] = np.nanmean(vec[0:stop+1])
        elif stop>len(vec):
            rm[i] = np.nanmean(vec[start:])
        else:
            rm[i] = np.nanmean(vec[start:stop+1])
    return rm

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
    plt.figure(figsize=(3.25,3))
    #plt.scatter(x, y, color='gainsboro', marker='.', alpha=0.1, zorder=0)
    avgs_x = np.unique(x)
    plt.scatter(avgs_x, avgs_y, facecolor='cornflower', edgecolor='black', linewidth=1.5, marker='o', s=60, zorder=5)
    if (len(subset)==2) | (len(subset)==5): # model only
        plt.axvline(np.nanmean(x), c='silver', linewidth=0.5, zorder=0)
        plt.axhline(np.nanmean(y), c='silver', linewidth=0.5, zorder=0.5)
    #plt.ylim([ylim[0], np.nanmax(y)])
    plt.ylim(ylim)

    if (ystr=='diff'):
        plt.axhline(0, c='k', linewidth=0.5, zorder=0)
        plt.axhspan(0, max(y), alpha=0.3, facecolor='deepskyblue', zorder=0)
        plt.axhspan(min(y), 0, alpha=0.2, facecolor='lightcoral', zorder=0)

    if metric=='hist_int':
        if ystr=='forecast':
            plt.ylabel('Histogram intersection \n(Average forecast)')
        elif ystr=='diff':
            plt.ylabel('Degradation')
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
    plt.savefig('../../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '.pdf') if len(xvar)==0 else plt.savefig('../../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_' + xvar + '.pdf')
    plt.close()
    return xfit[inds], pred_y_best_fit, rmse

def plot_heatmap_x_dimensionality_y(df, metric='hist_int', ystr='forecast', ylim=[0,1], subset='', var='NEE', xvar=''):
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
    plt.hist2d(x[keep], y[keep], bins=(40,80), cmap=LinearSegmentedColormap.from_list("", ["white","gainsboro","gray"]), alpha=0.9, cmin=1)
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

def plot_shading_x_dimensionality_y(df, metric='hist_int', ystr='forecast', ylim=[0,1], subset='', var='NEE', xvar=''):
    xstr = 'dimensionality' if len(xvar)==0 else xvar
    x = df[xstr].values

    if (ystr=='diff'):
        y = df[metric + '_forecast'].values - df[metric + '_calibration'].values
        avgs_y = df.groupby(xstr).median()[metric + '_forecast'].values - df.groupby(xstr).median()[metric + '_calibration'].values
        avgs_y_75 = df.groupby(xstr).quantile(.75)[metric + '_forecast'].values - df.groupby(xstr).quantile(.75)[metric + '_calibration'].values
        avgs_y_25 = df.groupby(xstr).quantile(.25)[metric + '_forecast'].values - df.groupby(xstr).quantile(.25)[metric + '_calibration'].values
        avgs_y_95 = df.groupby(xstr).quantile(.95)[metric + '_forecast'].values - df.groupby(xstr).quantile(.95)[metric + '_calibration'].values
        avgs_y_5 = df.groupby(xstr).quantile(.05)[metric + '_forecast'].values - df.groupby(xstr).quantile(.05)[metric + '_calibration'].values
    else:
        y = df[metric + '_' + ystr].values
        avgs_y = df.groupby(xstr).median()[metric + '_' + ystr].values
        avgs_y_75 = df.groupby(xstr).quantile(.75)[metric + '_' + ystr].values
        avgs_y_25 = df.groupby(xstr).quantile(.25)[metric + '_' + ystr].values
        avgs_y_95 = df.groupby(xstr).quantile(.95)[metric + '_' + ystr].values
        avgs_y_5 = df.groupby(xstr).quantile(.05)[metric + '_' + ystr].values
        avgs_y[df.groupby(xstr).count()[metric + '_' + ystr].values < 4] = float('nan')

    avgs_x = np.unique(x)
    plt.figure(figsize=(4,4))
    plt.fill_between(avgs_x, avgs_y_5, avgs_y_95, facecolor='whitesmoke', edgecolor='', alpha=0.6)
    plt.fill_between(avgs_x, avgs_y_25, avgs_y_75, facecolor='gainsboro', edgecolor='', alpha=0.9)
    plt.scatter(avgs_x, avgs_y, facecolor='dodgerblue', edgecolor='black', linewidth=1, marker='s', s=25, zorder=5)
    plt.ylim(ylim)
    plt.xlim([1,40]) if subset=='' else plt.xlim([1,40])

    if metric=='hist_int':
        if ystr=='forecast':
            plt.ylabel('Average forecast skill')
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

    plt.xlabel('Effective complexity') if len(xvar)==0 else plt.xlabel(xvar)
    plt.tight_layout()
    plt.savefig('../../plots/heatmaps/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_heatmap.pdf') if len(xvar)==0 else plt.savefig('../../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_' + xvar + '.pdf')
    plt.close()
    return

def plot_density_x_dimensionality_y(df, metric='hist_int', ystr='forecast', ylim=[0,1], subset='', var='NEE', xvar=''):
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

    xy = np.vstack([xk,yk])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x_density, y_density, z_density = xk[idx], yk[idx], z[idx]
    plt.scatter(x_density, y_density, c=z_density, edgecolor='',
        cmap=LinearSegmentedColormap.from_list("", ["white","whitesmoke","gainsboro","gray","dimgray"]), marker='o', alpha=0.25, s=20)

    avgs_x = np.unique(x)

    plt.scatter(avgs_x, avgs_y, facecolor='dodgerblue', edgecolor='black', linewidth=1.5, marker='o', s=50, zorder=5)
    plt.ylim(ylim)
    plt.xlim([1,41]) if subset=='' else plt.xlim([1,40])

    if metric=='hist_int':
        if ystr=='forecast':
            plt.ylabel('Average forecast skill')
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
    plt.savefig('../../plots/heatmaps/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_density.pdf') if len(xvar)==0 else plt.savefig('../../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_' + xvar + '.pdf')
    plt.close()
    return

def plot_scatter_x_dimensionality_y_colors(df, metric='hist_int', ystr='forecast', xstr='dimensionality', col_by='prior_minus_post', ylim=[0,1], subset='', var='NEE'):
    x = df[xstr].values

    if (ystr=='diff'):
        y = df[metric + '_forecast'].values - df[metric + '_calibration'].values
    else:
        y = df[metric + '_' + ystr].values

    p = np.random.permutation(len(x))
    col = df[col_by].values
    col_min = min(col)
    col_max = max(col)

    plt.figure(figsize=(6,5))
    sc_cbar = plt.scatter([x[p[0]]], [y[p[0]]], c=[col[p[0]]], cmap=plt.cm.gnuplot_r, marker='.', edgecolor='white',
        alpha=1, s=80, linewidth=0.25, vmin=col_min, vmax=col_max, zorder=0)
    plt.scatter(x[p], y[p], c=col[p], cmap=plt.cm.gnuplot_r, marker='.', edgecolor='white', alpha=0.7,
        s=80, linewidth=0.25,vmin=col_min, vmax=col_max, zorder=0)

    plt.ylim(ylim)
    if metric=='hist_int':
        plt.ylabel('histogram overlap (%s)' % ystr)
    elif metric=='RMSE':
        plt.ylabel('RMSE (%s)' % ystr)
    else:
        plt.ylabel(metric + '_' + ystr)

    cbar = plt.colorbar(sc_cbar, drawedges=False)
    cbar.set_label(col_by)

    plt.xlabel(xstr)
    plt.tight_layout()
    plt.savefig('../../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_' + xstr + '_col_by' + '_' + col_by + '.pdf')
    plt.close()
    return

def plot_scatter_x_dimensionality_y_resampled(df, metric='hist_int', ystr='forecast', ylim=[0,1], subset='', var='NEE'):
    grp_size = 20
    grps = df.groupby('dimensionality')

    xstr = 'dimensionality'
    x = df[xstr].values
    avgs_y_75 = df.groupby(xstr).quantile(.75)[metric + '_' + ystr].values
    avgs_y_25 = df.groupby(xstr).quantile(.25)[metric + '_' + ystr].values
    avgs_y_95 = df.groupby(xstr).quantile(.95)[metric + '_' + ystr].values
    avgs_y_5 = df.groupby(xstr).quantile(.05)[metric + '_' + ystr].values

    avgs_y_bootstrap = np.ones((1000, grps.ngroups))*np.nan
    for iter in range(1000):
        print(iter)
        grps_resampled = grps.apply(lambda x: x.sample(grp_size, replace=True))

        x = []
        y = []
        avgs_x = []
        avgs_y = []

        for i in range(int(df['dimensionality'].min()), int(df['dimensionality'].max()+1)):
            x.extend(grps_resampled.loc[i]['dimensionality'].values)
            avgs_x.extend([grps_resampled.loc[i]['dimensionality'].median()])

            if (ystr=='diff'):
                y.extend(grps_resampled.loc[i][metric + '_forecast'].values - grps_resampled.loc[i][metric + '_calibration'].values)
                avgs_y.extend([grps_resampled.loc[i][metric + '_forecast'].median() - grps_resampled.loc[i][metric + '_calibration'].median()])
            else:
                y.extend(grps_resampled.loc[i][metric + '_' + ystr].values)
                avgs_y.extend([grps_resampled.loc[i][metric + '_' + ystr].median()])
        avgs_y_bootstrap[iter,:] = avgs_y

    plt.figure(figsize=(4,4))
    plt.fill_between(avgs_x, avgs_y_5, avgs_y_95, facecolor='whitesmoke', edgecolor='', alpha=0.6)
    plt.fill_between(avgs_x, avgs_y_25, avgs_y_75, facecolor='gainsboro', edgecolor='', alpha=0.9)
    #plt.scatter(x, y, color='gainsboro', marker='.', alpha=0.4)
    #plt.scatter(avgs_x, avgs_y, facecolor='cornflowerblue', edgecolor='black', linewidth=1.5, marker='o', s=60)
    plt.scatter(avgs_x, np.nanmean(avgs_y_bootstrap, axis=0), facecolor='dodgerblue', edgecolor='black', linewidth=1.5, marker='o', s=50, zorder=2)
    plt.errorbar(avgs_x, np.nanmean(avgs_y_bootstrap, axis=0), yerr=np.nanstd(avgs_y_bootstrap, axis=0), c='k', linestyle='none', zorder=1)
    if (len(subset)==2) | (len(subset)==5): # model only
        plt.axvline(np.nanmean(x), c='silver', linewidth=0.5, zorder=0)
        plt.axhline(np.nanmean(y), c='silver', linewidth=0.5, zorder=0.5)
    #plt.ylim([ylim[0], np.nanmax(y)])
    plt.ylim(ylim)
    if metric=='hist_int':
        plt.ylabel('Average forecast skill')
    else:
        plt.ylabel(metric + '_' + ystr)
    plt.xlabel('Effective complexity')
    plt.tight_layout()
    plt.savefig('../../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_resampled.pdf')
    plt.close()
    return

def plot_scatter_model_avg_x_dimensionality_y(lx, ly, ey, ystr='hist_int_forecast', ylim=[0,1], var='NEE'):
    plt.figure(figsize=(5,5))
    plt.scatter(lx, ly, facecolor='cornflowerblue', edgecolor='black', linewidth=1.5, marker='o', s=60)
    plt.errorbar(lx, ly, yerr=ey, c='gainsboro', linestyle='none', alpha=0.4, zorder=0)
    plt.ylabel(ystr)
    plt.xlabel('number of model parameters')
    plt.savefig('../../plots/scatters/dimensionality/' + var + '/' + 'model_avg_' + ystr + '.pdf')
    plt.close()
    return

def plot_scatter_x_performance_y(df, xstr='calibration', ystr='forecast', metric='hist_int', subset='', var='NEE'):
    x = df[metric + '_' + xstr].values
    plt.figure(figsize=(5,5))
    if (ystr=='diff'):
        y = df[metric + '_forecast'].values - df[metric + '_calibration'].values
        plt.axhline(0, c='k', linewidth=0.5)
    else:
        y = df[metric + '_' + ystr].values
        plt.plot((0,1), c='k', linewidth=0.5)
    plt.scatter(x, y, c='mediumblue', s=15)
    plt.ylabel(metric + '_' + ystr)
    plt.xlabel(metric + '_' + xstr)
    plt.savefig('../../plots/scatters/performance/' + var + '/' + metric + '_' + xstr + '_' + metric + '_' + ystr + '_' + subset + '.pdf')
    plt.close()
    return

def plot_scatter_x_performance_y_dimensionality(df, list_of_subsets, xstr='calibration', ystr='forecast', metric='hist_int', subset='', var='NEE'):
    fig, ax = plt.subplots(1, len(list_of_subsets), figsize=(6*len(list_of_subsets),5))

    col_max = df['dimensionality'].max()
    col_min = 0

    sub_count = 0
    for el in list_of_subsets:
        if el=='nee':
            df_el = df[df['nee']==1]

        elif el=='no_nee':
            df_el = df[df['nee']==0]

        elif el=='nee_EDC':
            to_plot = subset_df_by_substring(df, '_EDC')
            df_el = df.loc[to_plot]
            df_el = df_el[df_el['nee']==1]

        elif el=='no_nee_noEDC':
            to_plot = subset_df_by_substring(df, '_noEDC')
            df_el = df.loc[to_plot]
            df_el = df_el[df_el['nee']==0]

        elif len(el)==1:
            df_el = df[df.index.str.endswith(el)]

        else:
            to_plot = subset_df_by_substring(df, el)
            df_el = df.loc[to_plot]

        x = df_el[metric + '_' + xstr].values
        col = df_el['dimensionality'].values

        if (ystr=='diff'):
            y = df_el[metric + '_forecast'].values - df_el[metric + '_calibration'].values
        else:
            y = df_el[metric + '_' + ystr].values

        p = np.random.permutation(len(x))
        sc_cbar = ax[sub_count].scatter([x[p[0]]], [y[p[0]]], c=[col[p[0]]], cmap=plt.cm.gist_stern_r,
            vmin=col_min, vmax=col_max, edgecolor='white', s=75, alpha=1, zorder=0)
        sc = ax[sub_count].scatter(x[p], y[p], c=col[p], cmap=plt.cm.gist_stern_r,
            vmin=col_min, vmax=col_max, edgecolor='white', s=100, alpha=0.6, zorder=1) #color='gainsboro',

        if metric=='hist_int':
            ax[sub_count].set_ylabel('histogram overlap (%s)' % ystr)
            ax[sub_count].set_xlabel('histogram overlap (%s)' % xstr)
        else:
            ax[sub_count].set_ylabel(metric + '_' + ystr)
            ax[sub_count].set_xlabel(metric + '_' + xstr)

        #plt.legend(loc='best', frameon=False)
        ax[sub_count].set_title(el.replace('_',''))

        if (ystr=='diff'):
            ax[sub_count].axhline(0, c='k', linewidth=0.5, zorder=1)
        else:
            print('average degradation = ')
            print(np.nanmean(y[p] - x[p]))
            print('percent positive = ')
            run_sum = 0
            total = 0
            for i in range(len(x)):
                xi = x[p[i]]
                yi = y[p[i]]
                if (yi - xi) > 0:
                    run_sum += 1
                total += 1
            print(run_sum/total * 100.)
            m, b = np.polyfit(x[p], y[p], 1)
            ax[sub_count].plot(np.linspace(0,1,100), m*np.linspace(0,1,100) + b, c='k', linewidth=1, zorder=1)
            ax[sub_count].plot((0,1), c='k', linewidth=0.5, linestyle='--', zorder=1)
            ax[sub_count].text(0.02, 0.74, 'average degradation = %.3f\npercent above 1:1 line = %.2f%%' % (np.nanmean(y[p] - x[p]), run_sum/total * 100.))
            ax[sub_count].set_xlim([0, 0.8])
            ax[sub_count].set_ylim([0, 0.8])

        cbar = fig.colorbar(sc_cbar, ax=ax[sub_count], drawedges=False)
        cbar.set_label('dimensionality')
        sub_count += 1

    plt.tight_layout()
    plt.savefig('../../plots/scatters/performance/' +  var + '/' + 'col_dimensionality_' + metric + '_' + xstr + '_' + metric + '_' + ystr + '_' + subset + '.pdf')
    plt.close()
    return

def plot_scatter_x_process_y_skill(df, process='', ystr='forecast', metric='hist_int', subset='', var='NEE'):
    process_values = np.unique(df[process].dropna().values)
    plt.figure(figsize=(5/4*len(process_values),5))
    sns.set_style('white')
    ax = sns.boxplot(x=process, y=metric+'_'+ystr, data=df, width=0.6, linewidth=0.75, palette=sns.light_palette('royalblue',len(process_values)),
        flierprops=dict(marker='.', markerfacecolor='k', markersize=3, markeredgecolor=None))
    medians = df.groupby([process])[metric+'_'+ystr].median()
    plt.setp(ax.artists, edgecolor='k', s=40)
    plt.setp(ax.lines, color='k')
    plt.axes().yaxis.grid(zorder=0, color='gainsboro', alpha=0.5)
    plt.ylim([0,None])
    plt.tight_layout()
    plt.savefig('../../plots/processes/' + var + '/' + process + '_' + metric + '_' + ystr + '_' + subset + '.pdf')
    plt.close()
    return medians

def plot_scatter_x_maxdiff_y_process(max_diff, processes, ystr='forecast', metric='hist_int', subset='', var='NEE'):
    max_diff, processes = zip(*sorted(zip(max_diff, processes)))
    plt.figure(figsize=(7,6))
    plt.barh(processes, max_diff, height=0.5, color='lightsteelblue', edgecolor='black', linewidth=0.75)#scatter(max_diff, np.arange(len(max_diff)))
    plt.xlabel('Maximum difference in median forecast skill between classes')
    plt.tight_layout()
    plt.savefig('../../plots/processes/' + var + '/max_diff_' + metric + '_' + ystr + '_' + subset + '.pdf')
    plt.close()
    return

def run_plots(df, subset_str='', var='NEE'):
    print(subset_str)
    plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='forecast', ylim=[0,0.8], subset=subset_str, var=var)
    plot_scatter_x_dimensionality_y(df, metric='RMSE', ystr='forecast', ylim=[0,3], subset=subset_str, var=var)

    try:
        plot_shading_x_dimensionality_y(df, metric='hist_int', ystr='forecast', ylim=[0,0.65], subset=subset_str, var=var)
        plot_scatter_x_dimensionality_y_resampled(df, metric='hist_int', ystr='forecast', ylim=[0,0.65], subset=subset_str, var=var)

    except Exception as e:
        print(e)

    #plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='diff', subset=subset_str, ylim=[-0.3,0.3], var=var)

    '''plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='forecast', subset=subset_str, ylim=[0,0.8], var=var, xvar='prior_minus_post')
    plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='forecast', subset=subset_str, ylim=[0,0.8], var=var, xvar='prior_minus_post_normalized')

    plot_scatter_x_dimensionality_y_colors(df, metric='hist_int', ystr='forecast', xstr='dimensionality', col_by='prior_minus_post', ylim=[0,0.8], subset=subset_str, var=var)
    plot_scatter_x_dimensionality_y_colors(df, metric='hist_int', ystr='forecast', xstr='dimensionality', col_by='prior_minus_post_normalized', ylim=[0,0.8], subset=subset_str, var=var)
    plot_scatter_x_dimensionality_y_colors(df, metric='hist_int', ystr='forecast', xstr='prior_minus_post', col_by='dimensionality', ylim=[0,0.8], subset=subset_str, var=var)
    plot_scatter_x_dimensionality_y_colors(df, metric='hist_int', ystr='forecast', xstr='dimensionality', col_by='prior_dim', ylim=[0,0.8], subset=subset_str, var=var)
    plot_scatter_x_dimensionality_y_colors(df, metric='hist_int', ystr='forecast', xstr='prior_minus_post', col_by='prior_dim', ylim=[0,0.8], subset=subset_str, var=var)'''

    return

def plot_density(df, subset_main='', subset_sub='', title='', subset_sub_list=[], var='NEE'):
    plt.figure(figsize=(8,5))
    to_plot_main = subset_df_by_substring(df, subset_main)
    sns.distplot(df.loc[to_plot_main]['dimensionality'],
        hist=False, kde=True, color='k', kde_kws={'bw':1, 'linewidth':2}, label='all data')
    #plt.axvline(df.loc[to_plot_main]['dimensionality'].mean(), color='k', linestyle='--', linewidth=0.5)
    if len(subset_sub_list)==0:
        if 'let_exp' in title:
            for sub in subset_sub:
                to_plot_sub = [str for str in to_plot_main if str.endswith(sub)]
                sns.distplot(df.loc[to_plot_sub]['dimensionality'],
                    hist=False, kde=True, kde_kws={'bw':1, 'linewidth':1, 'zorder':0}, label=sub.replace('_',''))
                #plt.axvline(df.loc[to_plot_sub]['dimensionality'].mean(), color=plt.gca().get_lines()[-1].get_c(), linestyle='--', linewidth=0.5)
        else:
            for sub in subset_sub:
                to_plot_sub = subset_list_by_substring(to_plot_main, sub)
                sns.distplot(df.loc[to_plot_sub]['dimensionality'],
                    hist=False, kde=True, kde_kws={'bw':1, 'linewidth':1, 'zorder':0}, label=sub.replace('_',''))
                #plt.axvline(df.loc[to_plot_sub]['dimensionality'].mean(), color=plt.gca().get_lines()[-1].get_c(), linestyle='--', linewidth=0.5)
    else:
        lbls = ['nee', 'no nee']
        count = 0
        for el in subset_sub_list:
            to_plot_sub = subset_list_by_list_of_substrings(to_plot_main, el)
            sns.distplot(df.loc[to_plot_sub]['dimensionality'],
                hist=False, kde=True, kde_kws={'bw':1, 'linewidth':1.5, 'zorder':0}, label=lbls[count])
            #plt.axvline(df.loc[to_plot_sub]['dimensionality'].mean(), color=plt.gca().get_lines()[-1].get_c(), linestyle='--', linewidth=0.5)
            count += 1
    plt.ylabel('Density')
    plt.ylim([0,None])
    plt.legend(loc='best', frameon=False, prop={'size': 8})
    plt.title(subset_main)
    plt.tight_layout()
    plt.savefig('../../plots/dists/' + var + '/' + subset_main + '_' + title + '.pdf')
    plt.close()
    return

def plot_dimensionality_medians(df, subset_main='', subset_sub='', title='', subset_sub_list=[], var='NEE', zero_point='default', fractional=False):
    plt.figure(figsize=(4.25,6.5))

    pars = raw_complexity()

    model_count = 0
    for i in subset_main:
        to_plot_main = subset_df_by_substring(df, i)

        if zero_point=='default':
            normalize_val_diff = 0
            normalize_val_coef = -1
            normalize_val_div = 1
        elif zero_point=='npars':
            normalize_val_diff = pars[pars['models']==i]['npars']
            normalize_val_coef = 1
            normalize_val_div = normalize_val_diff/100 if fractional else 1
        elif zero_point=='prior':
            normalize_val_diff = df.loc[[str for str in to_plot_main if 'noEDC_exp1f' in str]]['dimensionality'].mean()
            normalize_val_coef = 1
            normalize_val_div = normalize_val_diff/100 if fractional else 1

        sns.set_palette(sns.color_palette("Set2", len(subset_sub)))
        #plt.scatter(df.loc[to_plot_main]['dimensionality'].mean(), model_count, color='white', edgecolor='k', s=80, alpha=0.85, label='Mean of all runs' if model_count==len(subset_main)-1 else "")

        if zero_point!='prior':
            plt.scatter((normalize_val_diff - normalize_val_coef*pars[pars['models']==i]['npars'])/normalize_val_div, model_count, marker='*', color='white', edgecolor='k', s=80, label='Number of parameters' if model_count==len(subset_main)-1 else "")

        #plt.scatter((normalize_val_diff - normalize_val_coef*df.loc[[str for str in to_plot_main if 'noEDC_exp1f' in str]]['dimensionality'].mean())/normalize_val_div, model_count, marker='D', color='white', edgecolor='k', s=50, label='Prior' if model_count==len(subset_main)-1 else "")

        if len(subset_sub_list)==0:
            if 'let_exp' in title:
                switcher = {'a' : 'NEE',
                            'b' : 'NEE+LAI',
                            'c' : 'NEE+LAI+biomass',
                            'd' : 'LAI',
                            'e' : 'LAI+biomass',
                            'f' : 'No obs'
                }
                for sub in subset_sub:
                    to_plot_sub = [str for str in to_plot_main if str.endswith(sub)]
                    plt.scatter((normalize_val_diff - normalize_val_coef*df.loc[to_plot_sub]['dimensionality'].mean())/normalize_val_div, model_count, edgecolor='k', s=50, alpha=0.8, label=switcher[sub].replace('_','') + ' runs' if model_count==len(subset_main)-1 else "")
            elif 'num_exp' in title:
                switcher = {'exp1' : '50% error',
                            'exp2' : '100% error',
                            'exp3' : '150% error',
                            'exp4' : '200% error'
                }
                for sub in subset_sub:
                    to_plot_sub = subset_list_by_substring(to_plot_main, sub)
                    plt.scatter((normalize_val_diff - normalize_val_coef*df.loc[to_plot_sub]['dimensionality'].mean())/normalize_val_div, model_count, edgecolor='k', s=50, alpha=0.8, label=switcher[sub].replace('_','') + ' runs' if model_count==len(subset_main)-1 else "")

            else:
                for sub in subset_sub:
                    to_plot_sub = subset_list_by_substring(to_plot_main, sub)
                    plt.scatter((normalize_val_diff - normalize_val_coef*df.loc[to_plot_sub]['dimensionality'].mean())/normalize_val_div, model_count, edgecolor='k', s=50, alpha=0.8, label=sub.replace('_','') + ' runs' if model_count==len(subset_main)-1 else "")

        plt.axhline(y=model_count+0.5, color='darkgray', linewidth=0.5)

        model_count += 1

    plt.ylim([-0.5,model_count-0.5])
    plt.xlim([6,44]) if zero_point=='default' else plt.xlim([None,None])
    plt.ylabel('Model')
    if zero_point=='default':
        plt.xlabel('Effective complexity')
    else:
        if fractional:
            plt.xlabel('Dimensions reduced, normalized to prior (%)')
        else:
            plt.xlabel('Dimensions reduced\n(Prior minus posterior)')
    plt.yticks(np.arange(model_count), subset_main)
    loc = 'lower right' if zero_point=='default' else 'best'
    plt.legend(loc=loc, facecolor='white', edgecolor='black', framealpha=1, markerscale=0.85, prop={'size': 9.5})
    plt.tight_layout()
    plt.savefig('../../plots/dists/' + var + '/summary_' + title + '_' + zero_point + '_' + str(fractional) + '.pdf')
    plt.close()
    return

def plot_dimensionality_reduction_bar(df, subset_main='', title='', var='NEE', type=''):
    plt.figure(figsize=(4.25,6.5)) if len(subset_main)>1 else plt.figure(figsize=(2,2.5))
    model_count = 0

    cmap = matplotlib.cm.get_cmap('Set2')

    legend_labels = []
    for i in subset_main:
        to_plot_main = subset_df_by_substring(df, i)

        normalize_val_diff = df.loc[[str for str in to_plot_main if 'noEDC_exp1f' in str]]['dimensionality'].mean() if type!='dimensionality' else 0
        normalize_val_div = normalize_val_diff/100 if (type!='constrainability') & (type!='dimensionality') else 1

        EDC = ['_EDC', '_noEDC']
        EDC_spread = []
        for con in EDC:
            to_plot_sub = subset_list_by_substring(to_plot_main, con)
            EDC_spread.append((df.loc[to_plot_sub]['dimensionality'].mean() - normalize_val_diff)/normalize_val_div)
        legend_labels.append('EDCs')

        letter_experiments = ['a', 'b', 'c', 'd', 'e', 'f']
        let_spread = []
        for let in letter_experiments:
            to_plot_sub = [str for str in to_plot_main if str.endswith(let)]
            let_spread.append((df.loc[to_plot_sub]['dimensionality'].mean() - normalize_val_diff)/normalize_val_div)
        legend_labels.append('Assimilated data')

        numeric_experiments = get_experiments(type='numeric')
        num_spread = []
        for num in numeric_experiments:
            to_plot_sub = subset_list_by_substring(to_plot_main, num)
            num_spread.append((df.loc[to_plot_sub]['dimensionality'].mean() - normalize_val_diff)/normalize_val_div)
        legend_labels.append('Error scalar')

        sites = ['AU-How', 'FI-Hyy', 'FR-LBr', 'FR-Pue', 'GF-Guy', 'US-Ha1']
        site_spread = []
        for site in sites:
            to_plot_sub = subset_list_by_substring(to_plot_main, site)
            site_spread.append((df.loc[to_plot_sub]['dimensionality'].mean() - normalize_val_diff)/normalize_val_div)
        legend_labels.append('Sites')

        if len(subset_main)==1:
            models = raw_complexity().sort_values('npars')['models']
            model_spread = []
            for model in models:
                to_plot_sub = subset_list_by_substring(to_plot_main, model)
                model_spread.append((df.loc[to_plot_sub]['dimensionality'].mean() - normalize_val_diff)/normalize_val_div)
            legend_labels.append('Models')

            h = plt.barh(y=[model_count-0.333, model_count-0.167, model_count, model_count+0.167, model_count+0.333], width=[max(EDC_spread)-min(EDC_spread),
                max(let_spread)-min(let_spread), max(num_spread)-min(num_spread), max(site_spread)-min(site_spread),
                max(model_spread)-min(model_spread)], height=0.15, edgecolor='w', linewidth=0.0, color=[cmap(0), cmap(1), cmap(2), cmap(3), cmap(4)])
            plt.yticks([])

        else:
            if title=='bar':
                h = plt.barh(y=[model_count-0.2, model_count-0.0667, model_count+0.0667, model_count+0.2], width=[max(EDC_spread)-min(EDC_spread),
                    max(let_spread)-min(let_spread), max(num_spread)-min(num_spread), max(site_spread)-min(site_spread)], height=0.1,
                    edgecolor='w', linewidth=0.0, color=[cmap(0), cmap(1), cmap(2), cmap(3)])
            else:
                plt.plot([min(EDC_spread), max(EDC_spread)], [model_count-0.3, model_count-0.3], linewidth=5, color=cmap(0))
                plt.plot([min(let_spread), max(let_spread)], [model_count-0.1, model_count-0.1], linewidth=5, color=cmap(1))
                plt.plot([min(num_spread), max(num_spread)], [model_count+0.1, model_count+0.1], linewidth=5, color=cmap(2))
                plt.plot([min(site_spread), max(site_spread)], [model_count+0.3, model_count+0.3], linewidth=5, color=cmap(3))

        plt.axhline(y=model_count+0.5, color='darkgray', linewidth=0.5, label='_nolegend_')

        model_count += 1

    plt.ylim([-0.5,model_count-0.5])
    plt.xlim([6,44]) if title!='bar_full' else plt.xlim([None, None])
    plt.ylabel('Model') if len(subset_main)>1 else plt.ylabel('')
    if type=='constrainability':
        plt.xlabel('Range of constrainability \n(number of dimensions)')
    elif type=='dimensionality':
        if title=='lines':
            plt.xlabel('Effective complexity')
        else:
            plt.xlabel('Attributed dimensionality range') if len(subset_main)>1 else plt.xlabel('Total complexity \nrange')
    else:
        plt.xlabel('Range of normalized dimensionality reduction (%)')

    plt.yticks(np.arange(model_count), subset_main)
    if (title!='lines') & (len(subset_main)>1):
        plt.legend(h, legend_labels, loc='best', facecolor='white', edgecolor='black', framealpha=1, prop={'size': 9})
    plt.tight_layout()
    plt.savefig('../../plots/dists/' + var + '/summary_' + title + '_' + type + '.pdf')
    plt.close()

def compute_statistics_of_fit(avgs_x, pred_y):
    if np.isscalar(pred_y):
        peak_location = float('nan')
        range_y = float('nan')
        max_skill = float('nan')
        slope = float('nan')
    else:
        peak_location = avgs_x[np.argmax(pred_y)]
        range_y = pred_y[-1] - pred_y[0]
        max_skill = np.nanmax(pred_y)
        slope = range_y/(avgs_x[-1]-avgs_x[0])
    return peak_location, range_y, max_skill, slope

def plot_statistics_of_fit_twoaxes(peak_locations, range_ys, labels, obs_binary):
    fig, ax = plt.subplots(2,2,figsize=(7,6.5))

    colors = ['white','white','dodgerblue','dodgerblue','crimson','crimson','yellowgreen','yellowgreen','darkorange','darkorange',
        'dodgerblue','dodgerblue','crimson','crimson','yellowgreen','yellowgreen','darkorange','darkorange','mediumorchid','mediumorchid','gold','gold',
        'dodgerblue','dodgerblue','crimson','crimson','yellowgreen','yellowgreen','darkorange','darkorange','mediumorchid','mediumorchid','gold','gold',
        'dodgerblue','dodgerblue','crimson','crimson']

    for label in labels:
        if ' ' in label:
            for row in [0,1]:
                for col in [0,1]:
                    m = 'o' if obs_binary[labels.index(label)]==0 else '^'
                    ax[row,col].scatter(peak_locations[labels.index(label)], range_ys[labels.index(label)], s=125, c=colors[labels.index(label)], edgecolor='k', linewidth=0.5, marker=m, label=label)
                    ax[row,col].set_xlabel('Dimensionality at which \nmaximum skill is achieved')
                    ax[row,col].set_ylabel('Range in skill between highest\n and lowest dimensionality')
                    ax[row,col].set_xlim([5,40])
                    ax[row,col].set_ylim([-0.25,0.35])

        if (label[0]=='e') & (any(i.isdigit() for i in label)):
            m = 'o' if obs_binary[labels.index(label)]==0 else '^'
            l = labels[labels.index(label)] if obs_binary[labels.index(label)]==0 else labels[labels.index(label)][0:-4] + ' with obs'
            if obs_binary[labels.index(label)]==0:
                continue
            else:
                ax[0,1].scatter(peak_locations[labels.index(label)], range_ys[labels.index(label)], s=125, c=colors[labels.index(label)], edgecolor='k', linewidth=0.5, marker=m, label=l)
        ax[0,1].legend(loc='best', facecolor='white', edgecolor='k', framealpha=0.5, markerscale=0.5, prop={'size': 7})

        if (label[0]=='e') & ~(any(i.isdigit() for i in label)):
            m = 'o'
            l = label if obs_binary[labels.index(label)]==0 else '_nolegend_'
            ax[0,0].scatter(peak_locations[labels.index(label)], range_ys[labels.index(label)], s=125, c=colors[labels.index(label)], edgecolor='k', linewidth=0.5, marker=m, label=l)
        ax[0,0].legend(loc='best', facecolor='white', edgecolor='k', framealpha=0.5, markerscale=0.5, prop={'size': 7})

        if '-' in label:
            m = 'o' if obs_binary[labels.index(label)]==0 else '^'
            l = labels[labels.index(label)] if obs_binary[labels.index(label)]==0 else labels[labels.index(label)][0:-4] + ' with obs'
            if obs_binary[labels.index(label)]==0:
                continue
            else:
                ax[1,0].scatter(peak_locations[labels.index(label)], range_ys[labels.index(label)], s=125, c=colors[labels.index(label)], edgecolor='k', linewidth=0.5, marker=m, label=l)
        ax[1,0].legend(loc='best', facecolor='white', edgecolor='k', framealpha=0.5, markerscale=0.5, prop={'size': 7})

        if 'EDC' in label:
            m = 'o' if obs_binary[labels.index(label)]==0 else '^'
            l = labels[labels.index(label)] if obs_binary[labels.index(label)]==0 else labels[labels.index(label)][0:-4] + ' with obs'
            if obs_binary[labels.index(label)]==0:
                continue
            else:
                ax[1,1].scatter(peak_locations[labels.index(label)], range_ys[labels.index(label)], s=125, c=colors[labels.index(label)], edgecolor='k', linewidth=0.5, marker=m, label=l)
        ax[1,1].legend(loc='best', facecolor='white', edgecolor='k', framealpha=0.5, markerscale=0.5, prop={'size': 7})

    plt.tight_layout()
    plt.savefig('../../plots/scatters/dimensionality/summary_statistics_obs.pdf')
    plt.close()
    return

def plot_statistics_of_fit(peak_locations, range_ys, max_skills, slopes, rmses, labels, obs_binary):
    locations = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 2, 2, 3, 3]
    colors = ['white','white','midnightblue','midnightblue','mediumblue','mediumblue','royalblue','royalblue','cornflowerblue','cornflowerblue',
        'midnightblue','midnightblue','mediumblue','mediumblue','royalblue','royalblue','cornflowerblue','cornflowerblue','lightskyblue','lightskyblue','lightcyan','lightcyan',
        'midnightblue','midnightblue','mediumblue','mediumblue','royalblue','royalblue','cornflowerblue','cornflowerblue','lightskyblue','lightskyblue','lightcyan','lightcyan',
        'midnightblue','midnightblue','mediumblue','mediumblue']

    for labels_start, labels_end, subset, figwidth in [[2,9, 'error', 6.5],[10,21, 'data', 7.5],[22,33, 'site', 9],[34,37, 'EDC', 3.5]]:
        fig, ax = plt.subplots(1,3,figsize=(7.5,2.5))
        xtick = []
        xtickloc = []
        palette = sns.light_palette('royalblue',(labels_end+1-labels_start))

        count = 0
        for i in range(labels_start, labels_end+1):
            if obs_binary[i]==1:
                ax[0].bar(locations[i], peak_locations[i], facecolor=palette[count], edgecolor='k', width=0.5, linewidth=1)
                ax[1].bar(locations[i], range_ys[i], facecolor=palette[count], edgecolor='k', width=0.5, linewidth=1)
                ax[2].bar(locations[i], max_skills[i], facecolor=palette[count], edgecolor='k', width=0.5, linewidth=1)
                #ax[3].bar(locations[i], slopes[i], facecolor=palette[count], edgecolor='k', width=0.5, linewidth=1)
                xtick.append(labels[i][0:-4])
                xtickloc.append(locations[i])
            count += 1

        for j, metric, ylim in [0, 'Optimum\n dimensionality',[0,40]],[1,'Range of skill',[-0.025,0.28]],[2,'Maximum skill',[0,0.43]]:#,[3,'Slope',[-0.001,0.009]]:
            ax[j].axhline(y=0, c='k', linewidth=0.5)
            ax[j].set_xticks(xtickloc)
            ax[j].set_xticklabels(xtick, rotation=60)
            ax[j].set_ylabel(metric)
            ax[j].set_xlim([1.5,len(xtick)+1.5]) if labels_start!=10 else ax[j].set_xlim([1.5,len(xtick)+0.5])
            ax[j].set_ylim(ylim)
        plt.tight_layout()
        fig.subplots_adjust(wspace=.5)
        plt.savefig('../../plots/scatters/dimensionality/summary_statistics_' + subset + '.pdf')
        plt.close()

        plt.figure(figsize=(figwidth,3))
        count = 0
        xtick = []
        xtickloc = []
        palette = sns.light_palette('royalblue',(3))
        for i in range(labels_start, labels_end+1):
            if obs_binary[i]==1:
                barlist = plt.bar([locations[i]-0.25, locations[i], locations[i]+0.25], rmses[i], edgecolor='k', linewidth=1, width=0.15)
                for j in range(3):
                    barlist[j].set_facecolor(palette[j])

                xtick.append(labels[i][0:-4])
                xtickloc.append(locations[i])
            count += 1

        plt.xticks(xtickloc, xtick, rotation=60)
        plt.ylabel('RMSE')
        plt.xlim([1.5,len(xtick)+1.5]) if labels_start!=10 else plt.xlim([1.5,len(xtick)+0.5])
        plt.ylim([0,0.1])
        plt.legend(handles=[mpatches.Patch(facecolor=palette[0], edgecolor='k', label='Constant fit'),
            mpatches.Patch(facecolor=palette[1], edgecolor='k', label='Linear fit'),
            mpatches.Patch(facecolor=palette[2], edgecolor='k', label='Higher-order fit')], loc='upper left', facecolor='white', edgecolor='k', framealpha=1, prop={'size': 8})
        plt.tight_layout()
        plt.savefig('../../plots/scatters/dimensionality/summary_statistics_fits_' + subset + '.pdf')
        plt.close()

    return
