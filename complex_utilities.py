
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
    data = [['C1',23],['C2',33],['C3',35],['C4',34],['C5',34],['C6',23],['C7',27],['C8',36],['E1',17],
        ['G1',37],['G2',40],['G3',43],['G4',43],['S1',11],['S2',14],['S4',17]]
    return DataFrame(data,columns=['models','npars'])

def processes_discrete(model=None):
    data = [['C1',23.,np.nan,np.nan,np.nan,0.,0.,1.,1.,0.,0.,0.,2.,4.,0.,6.,0],
            ['C2',33.,np.nan,np.nan,np.nan,1.,1.,2.,2.,0.,0.,0.,2.,4.,1.,7.,1],
            ['C3',35.,np.nan,np.nan,np.nan,1.,1.,2.,2.,0.,0.,0.,2.,4.,1.,7.,1],
            ['C4',34.,np.nan,np.nan,np.nan,1.,1.,2.,2.,0.,0.,0.,2.,4.,1.,7.,1],
            ['C5',34.,np.nan,np.nan,np.nan,1.,1.,1.,2.,3.,2.,1.,np.nan,np.nan,np.nan,np.nan,np.nan],
            ['C6',23.,np.nan,np.nan,np.nan,0.,0.,1.,1.,0.,1.,2.,3.,4.,0.,np.nan,np.nan],
            ['C7',27.,np.nan,np.nan,np.nan,1.,0.,1.,1.,0.,1.,2.,3.,4.,5.,np.nan,np.nan],
            ['C8',36.,np.nan,np.nan,np.nan,1.,0.,2.,1.,0.,0.,2.,8.,1.,np.nan,np.nan,np.nan],
            ['E1',17.,np.nan,np.nan,np.nan,0.,0.,0.,0.,0.,0.,0.,3.,3.,0.,np.nan,np.nan],
            ['G1',37.,np.nan,np.nan,np.nan,0.,0.,2.,3.,1.,1.,2.,3.,4.,0.,np.nan,np.nan],
            ['G2',40.,np.nan,np.nan,np.nan,1.,0.,2.,3.,1.,1.,2.,3.,4.,3.,np.nan,np.nan],
            ['G3',43.,np.nan,np.nan,np.nan,0.,0.,2.,4.,2.,1.,2.,3.,4.,0.,np.nan,np.nan],
            ['G4',43.,np.nan,np.nan,np.nan,1.,0.,2.,4.,2.,1.,2.,3.,4.,3.,np.nan,np.nan],
            ['S1',11.,np.nan,np.nan,np.nan,0.,0.,0.,0.,0.,0.,0.,1.,2.,0.,np.nan,np.nan],
            ['S2',14.,np.nan,np.nan,np.nan,0.,0.,1.,1.,0.,0.,0.,3.,np.nan,np.nan,np.nan,np.nan],
            ['S4',17.,np.nan,np.nan,np.nan,0.,0.,1.,1.,0.,0.,0.,3.,2.,0.,np.nan,np.nan]]

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
    plt.xlabel('dimensionality') if len(xvar)==0 else plt.xlabel('prior minus posterior dimensionality')
    plt.tight_layout()
    plt.savefig('../../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '.pdf') if len(xvar)==0 else plt.savefig('../../plots/scatters/dimensionality/' + var + '/' + metric + '_' + ystr + '_' + subset + '_' + xvar + '.pdf')
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
    plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='forecast', subset=subset_str, ylim=[0,0.8], var=var)
    plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='diff', subset=subset_str, ylim=[-0.3,0.3], var=var)

    plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='forecast', subset=subset_str, ylim=[0,0.8], var=var, xvar='prior_minus_post')
    plot_scatter_x_dimensionality_y(df, metric='hist_int', ystr='forecast', subset=subset_str, ylim=[0,0.8], var=var, xvar='prior_minus_post_normalized')

    plot_scatter_x_dimensionality_y_colors(df, metric='hist_int', ystr='forecast', xstr='dimensionality', col_by='prior_minus_post', ylim=[0,0.8], subset=subset_str, var=var)
    plot_scatter_x_dimensionality_y_colors(df, metric='hist_int', ystr='forecast', xstr='dimensionality', col_by='prior_minus_post_normalized', ylim=[0,0.8], subset=subset_str, var=var)
    plot_scatter_x_dimensionality_y_colors(df, metric='hist_int', ystr='forecast', xstr='prior_minus_post', col_by='dimensionality', ylim=[0,0.8], subset=subset_str, var=var)
    plot_scatter_x_dimensionality_y_colors(df, metric='hist_int', ystr='forecast', xstr='dimensionality', col_by='prior_dim', ylim=[0,0.8], subset=subset_str, var=var)
    plot_scatter_x_dimensionality_y_colors(df, metric='hist_int', ystr='forecast', xstr='prior_minus_post', col_by='prior_dim', ylim=[0,0.8], subset=subset_str, var=var)

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
    plt.figure(figsize=(7,7))

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
            plt.scatter((normalize_val_diff - normalize_val_coef*pars[pars['models']==i]['npars'])/normalize_val_div, model_count, marker='*', color='white', edgecolor='k', s=120, label='Number of parameters' if model_count==len(subset_main)-1 else "")

        plt.scatter((normalize_val_diff - normalize_val_coef*df.loc[[str for str in to_plot_main if 'noEDC_exp1f' in str]]['dimensionality'].mean())/normalize_val_div, model_count, marker='D', color='white', edgecolor='k', s=50, label='Prior' if model_count==len(subset_main)-1 else "")

        if len(subset_sub_list)==0:
            if 'let_exp' in title:
                for sub in subset_sub:
                    to_plot_sub = [str for str in to_plot_main if str.endswith(sub)]
                    plt.scatter((normalize_val_diff - normalize_val_coef*df.loc[to_plot_sub]['dimensionality'].mean())/normalize_val_div, model_count, edgecolor='k', s=80, alpha=0.85, label='Mean of ' + sub.replace('_','') + ' runs' if model_count==len(subset_main)-1 else "")
            else:
                for sub in subset_sub:
                    to_plot_sub = subset_list_by_substring(to_plot_main, sub)
                    plt.scatter((normalize_val_diff - normalize_val_coef*df.loc[to_plot_sub]['dimensionality'].mean())/normalize_val_div, model_count, edgecolor='k', s=80, alpha=0.85, label='Mean of ' + sub.replace('_','') + ' runs' if model_count==len(subset_main)-1 else "")

        plt.axhline(y=model_count+0.5, color='gainsboro', linewidth=0.5)

        model_count += 1

    plt.ylim([-0.5,model_count-0.5])
    plt.xlim([5,None]) if zero_point=='default' else plt.xlim([None,None])
    plt.ylabel('Model')
    if zero_point=='default':
        plt.xlabel('Dimensionality')
    else:
        if fractional:
            plt.xlabel('Dimensions reduced, normalized to prior (%)')
        else:
            plt.xlabel('Dimensions reduced\n(Prior minus posterior)')
    plt.yticks(np.arange(model_count), subset_main)
    loc = 'lower right' if zero_point=='default' else 'best'
    plt.legend(loc=loc, facecolor='white', edgecolor='black', framealpha=1, prop={'size': 9})
    plt.tight_layout()
    plt.savefig('../../plots/dists/' + var + '/summary_' + title + '_' + zero_point + '_' + str(fractional) + '.pdf')
    plt.close()
    return

def plot_dimensionality_reduction_bar(df, subset_main='', title='', var='NEE', type=''):
    plt.figure(figsize=(7,9)) if len(subset_main)>1 else plt.figure(figsize=(3,3))
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
        legend_labels.append('Observational error')

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
                max(model_spread)-min(model_spread)], height=0.15, edgecolor='k', linewidth=0.75, color=[cmap(0), cmap(1), cmap(2), cmap(3), cmap(4)])

        else:
            h = plt.barh(y=[model_count-0.3, model_count-0.1, model_count+0.1, model_count+0.3], width=[max(EDC_spread)-min(EDC_spread),
                max(let_spread)-min(let_spread), max(num_spread)-min(num_spread), max(site_spread)-min(site_spread)], height=0.15,
                edgecolor='k', linewidth=0.75, color=[cmap(0), cmap(1), cmap(2), cmap(3)])
        plt.axhline(y=model_count+0.5, color='lightgray', linewidth=0.5, label='_nolegend_')

        model_count += 1

    plt.ylim([-0.5,model_count-0.5])
    plt.ylabel('Model') if len(subset_main)>1 else plt.ylabel('')
    if type=='constrainability':
        plt.xlabel('Range of constrainability \n(number of dimensions)')
    elif type=='dimensionality':
        plt.xlabel('Range of posterior dimensionality')
    else:
        plt.xlabel('Range of normalized dimensionality reduction (%)')
    plt.yticks(np.arange(model_count), subset_main)
    plt.legend(h, legend_labels, loc='best', facecolor='white', edgecolor='black', framealpha=1, prop={'size': 9})
    plt.tight_layout()
    plt.savefig('../../plots/dists/' + var + '/summary_' + title + '_' + type + '.pdf')
    plt.close()
