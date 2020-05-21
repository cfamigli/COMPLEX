
import numpy as np
from pandas import read_csv, to_numeric, DataFrame
import glob
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

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
    data = [['C1',23],['C2',33],['C3',35],['C4',34],['C5',34],['C6',23],['C7',46],['C8',36],['E1',17],
        ['G1',37],['G2',40],['G3',43],['G4',43],['S1',11],['S2',14],['S4',17]]
    return DataFrame(data,columns=['models','npars'])

def processes_discrete(model=None):
    data = [['C1',0.,0.,1.,1.,0.,0.,0.,2.,4.,0.,6.,0],
            ['C2',1.,1.,2.,2.,0.,0.,0.,2.,4.,1.,7.,1],
            ['C3',1.,1.,2.,2.,0.,0.,0.,2.,4.,1.,7.,1],
            ['C4',1.,1.,2.,2.,0.,0.,0.,2.,4.,1.,7.,1],
            ['C5',1.,1.,1.,2.,3.,2.,1.,np.nan,np.nan,np.nan,np.nan,np.nan],
            ['C6',0.,0.,1.,1.,0.,1.,2.,3.,4.,0.,np.nan,np.nan],
            ['C7',1.,0.,1.,1.,0.,1.,2.,3.,4.,5.,np.nan,np.nan],
            ['C8',1.,0.,2.,1.,0.,0.,2.,8.,1.,np.nan,np.nan,np.nan],
            ['E1',0.,0.,0.,0.,0.,0.,0.,3.,3.,0.,np.nan,np.nan],
            ['G1',0.,0.,2.,3.,1.,1.,2.,3.,4.,0.,np.nan,np.nan],
            ['G2',1.,0.,2.,3.,1.,1.,2.,3.,4.,3.,np.nan,np.nan],
            ['G3',0.,0.,2.,4.,2.,1.,2.,3.,4.,0.,np.nan,np.nan],
            ['G4',1.,0.,2.,4.,2.,1.,2.,3.,4.,3.,np.nan,np.nan],
            ['S1',0.,0.,0.,0.,0.,0.,0.,1.,2.,0.,np.nan,np.nan],
            ['S2',0.,0.,1.,1.,0.,0.,0.,3.,np.nan,np.nan,np.nan,np.nan],
            ['S4',0.,0.,1.,1.,0.,0.,0.,3.,2.,0.,np.nan,np.nan]]

    return_df = DataFrame(data,columns=['model','PAW','Rh','labile_c_lifespan','phenology','CUE',
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
        R2 = r2_score(obs_data[~inds], pred_data[~inds])
        RMSE = np.sqrt(mean_squared_error(obs_data[~inds], pred_data[~inds]))
    except:
        R2 = float('nan')
        RMSE = float('nan')
    return (R2, RMSE)

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

def plot_time_series_with_spread(obs_data, pred_data, obs_unc, cal_period_stop, var='NEE', title=None):
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
        facecolor='lightcoral', alpha=0.8, label='observational uncertainty', zorder=1.5)
    plt.ylabel(var)
    plt.xlabel('months after start')
    plt.legend(loc='best')
    plt.savefig('../../../../plots/time_series_with_spread/' + title + '.pdf')
    plt.close()
    return

def plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='forecast', ylim=[0,1], subset='', var='NEE'):
    x = df['dimensionality'].values
    if (ystr=='diff'):
        y = df[metric + '_forecast'].values - df[metric + '_calibration'].values
        avgs_y = df.groupby('dimensionality').median()[metric + '_forecast'].values - df.groupby('dimensionality').median()[metric + '_calibration'].values
    else:
        y = df[metric + '_' + ystr].values
        avgs_y = df.groupby('dimensionality').median()[metric + '_' + ystr].values
    plt.figure(figsize=(5,5))
    plt.scatter(x, y, color='gainsboro', marker='.', alpha=0.4)
    avgs_x = np.unique(x)
    plt.scatter(avgs_x, avgs_y, facecolor='cornflowerblue', edgecolor='black', linewidth=1.5, marker='o', s=60)
    if (len(subset)==2) | (len(subset)==5): # model only
        plt.axvline(np.nanmean(x), c='silver', linewidth=0.5, zorder=0)
        plt.axhline(np.nanmean(y), c='silver', linewidth=0.5, zorder=0.5)
    #plt.ylim([ylim[0], np.nanmax(y)])
    plt.ylim(ylim)
    if metric=='hist_int':
        plt.ylabel('histogram overlap (%s)' % ystr)
    elif metric=='RMSE':
        plt.ylabel('RMSE (%s)' % ystr)
    else:
        plt.ylabel(metric + '_' + ystr)
    plt.xlabel('dimensionality')
    plt.tight_layout()
    plt.savefig('../../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '.pdf')
    plt.close()
    return

def plot_scatter_x_dimensionality_y_resampled(df, metric='hist_int', ystr='forecast', ylim=[0,1], subset='', var='NEE'):
    grp_size = 20
    grps = df.groupby('dimensionality')

    avgs_y_bootstrap = np.ones((1000, grps.ngroups))*np.nan
    for iter in range(1000):
        print(iter)
        grps_resampled = grps.apply(lambda x: x.sample(grp_size, replace=True))

        x = []
        y = []
        avgs_x = []
        avgs_y = []

        for i in range(int(grps.ngroups)):
            x.extend(grps_resampled.loc[i]['dimensionality'].values)
            avgs_x.extend([grps_resampled.loc[i]['dimensionality'].median()])

            if (ystr=='diff'):
                y.extend(grps_resampled.loc[i][metric + '_forecast'].values - grps_resampled.loc[i][metric + '_calibration'].values)
                avgs_y.extend([grps_resampled.loc[i][metric + '_forecast'].median() - grps_resampled.loc[i][metric + '_calibration'].median()])
            else:
                y.extend(grps_resampled.loc[i][metric + '_' + ystr].values)
                avgs_y.extend([grps_resampled.loc[i][metric + '_' + ystr].median()])
        avgs_y_bootstrap[iter,:] = avgs_y

    plt.figure(figsize=(5,5))
    plt.scatter(x, y, color='gainsboro', marker='.', alpha=0.4)
    #plt.scatter(avgs_x, avgs_y, facecolor='cornflowerblue', edgecolor='black', linewidth=1.5, marker='o', s=60)
    plt.scatter(avgs_x, np.nanmean(avgs_y_bootstrap, axis=0), facecolor='cornflowerblue', edgecolor='black', linewidth=1.5, marker='o', s=60, zorder=2)
    plt.errorbar(avgs_x, np.nanmean(avgs_y_bootstrap, axis=0), yerr=np.nanstd(avgs_y_bootstrap, axis=0), c='k', linestyle='none', zorder=1)
    if (len(subset)==2) | (len(subset)==5): # model only
        plt.axvline(np.nanmean(x), c='silver', linewidth=0.5, zorder=0)
        plt.axhline(np.nanmean(y), c='silver', linewidth=0.5, zorder=0.5)
    #plt.ylim([ylim[0], np.nanmax(y)])
    plt.ylim(ylim)
    if metric=='hist_int':
        plt.ylabel('histogram overlap (%s)' % ystr)
    else:
        plt.ylabel(metric + '_' + ystr)
    plt.xlabel('dimensionality')
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

def plot_scatter_x_performance_y(df, xstr='calibration', ystr='forecast',
    metric='hist_int', subset='', var='NEE'):
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

def plot_scatter_x_performance_y_multicolor(df, list_of_subsets, xstr='calibration', ystr='forecast',
    metric='hist_int', subset='', var='NEE'):
    plt.figure(figsize=(5,5))
    for el in list_of_subsets:
        to_plot = subset_df_by_substring(df, el)
        df_el = df.loc[to_plot]
        x = df_el[metric + '_' + xstr].values
        col = df_el['dimensionality'].values

        if (ystr=='diff'):
            y = df_el[metric + '_forecast'].values - df_el[metric + '_calibration'].values
        else:
            y = df_el[metric + '_' + ystr].values

        if 'nee' not in subset:
            sc = plt.scatter(x, y, edgecolor="None", marker='.', alpha=0.2, zorder=0) #color='gainsboro',
            plt.scatter(np.nanmean(x), np.nanmean(y), color=sc.get_facecolors(),
                edgecolor='black', linewidth=1.5, marker='o', s=100, alpha=1, label=el.replace('_',''))
        else:
            sc = plt.scatter(x, y, c='lightgray', edgecolor="None", marker='o', alpha=0.5, zorder=0) #color='gainsboro',
            plt.title(subset)

        if metric=='hist_int':
            plt.ylabel('histogram overlap (%s)' % ystr)
            plt.xlabel('histogram overlap (%s)' % xstr)
        else:
            plt.ylabel(metric + '_' + ystr)
            plt.xlabel(metric + '_' + xstr)

        plt.legend(loc='best', frameon=False)
    if (ystr=='diff'):
        plt.axhline(0, c='k', linewidth=0.5, zorder=0)
    else:
        plt.plot((0,1), c='k', linewidth=0.5, zorder=0)
        plt.xlim([0, 0.8])
        plt.ylim([0, 0.8])

    plt.savefig('../../plots/scatters/performance/' + var + '/' + 'multicolor_base_' + metric + '_' + xstr + '_' + metric + '_' + ystr + '_' + subset + '.pdf')
    plt.close()
    return

def plot_scatter_x_performance_y_dimensionality(df, list_of_subsets, xstr='calibration', ystr='forecast',
    metric='hist_int', subset='', var='NEE'):
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
    plt.figure(figsize=(5/2*len(process_values),5))
    sns.set_style('white')
    ax = sns.violinplot(x=process, y=metric+'_'+ystr, data=df, palette=sns.hls_palette(5, h=.5, l=.7),
        cut=0, linewidth=1, width=0.75, scale_hue=False)
    plt.axes().yaxis.grid(zorder=0, color='gainsboro')
    plt.ylim([0,None])
    plt.savefig('../../plots/scatters/processes/' + var + '/' + process + '_' + metric + '_' + ystr + '_' + subset + '.pdf')
    plt.close()
    return

def run_plots(df, subset_str='', var='NEE'):
    plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='forecast', subset=subset_str, ylim=[0,0.8], var=var)
    plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='diff', subset=subset_str, ylim=[-0.3,0.3], var=var)

    plot_scatter_x_dimensionality_y(df, metric='R2', ystr='forecast', subset=subset_str, ylim=[-1,1], var=var)
    plot_scatter_x_dimensionality_y(df, metric='R2', ystr='diff', subset=subset_str, ylim=[-0.3,1], var=var)

    plot_scatter_x_dimensionality_y(df, metric='RMSE', ystr='forecast', subset=subset_str, ylim=[0,5], var=var)
    plot_scatter_x_dimensionality_y(df, metric='RMSE', ystr='diff', subset=subset_str, ylim=[-0.5,1], var=var)


    '''if (subset_str=='') | (subset_str=='good_only'):
        plot_scatter_x_dimensionality_y_resampled(df, metric='hist_int', ystr='forecast', subset=subset_str, ylim=[0,0.6])
        plot_scatter_x_dimensionality_y_resampled(df, metric='hist_int', ystr='diff', subset=subset_str, ylim=[-0.3,0.3])'''
    return

def plot_density(df, subset_main='', subset_sub='', title='', subset_sub_list=[], var='NEE'):
    plt.figure(figsize=(5,5))
    to_plot_main = subset_df_by_substring(df, subset_main)
    sns.distplot(df.loc[to_plot_main]['dimensionality'],
        hist=False, kde=True, color='k', kde_kws={'bw':1, 'linewidth':2}, label='all data')
    if len(subset_sub_list)==0:
        for sub in subset_sub:
            to_plot_sub = subset_list_by_substring(to_plot_main, sub)
            sns.distplot(df.loc[to_plot_sub]['dimensionality'],
                hist=False, kde=True, kde_kws={'bw':1, 'linewidth':1}, label=sub.replace('_',''))
    else:
        lbls = ['nee', 'no nee']
        count = 0
        for el in subset_sub_list:
            to_plot_sub = subset_list_by_list_of_substrings(to_plot_main, el)
            sns.distplot(df.loc[to_plot_sub]['dimensionality'],
                hist=False, kde=True, kde_kws={'bw':1, 'linewidth':1}, label=lbls[count])
            count += 1
    plt.ylabel('density')
    plt.legend(loc='best')
    plt.title(subset_main)
    plt.tight_layout()
    plt.savefig('../../plots/dists/' + var + '/' + subset_main + '_' + title + '.pdf')
    plt.close()
    return
