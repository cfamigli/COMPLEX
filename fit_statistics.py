
import os
import sys
import numpy as np
from pandas import read_pickle, DataFrame
import complex_utilities as computil
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def main():
    os.chdir('../data/analysis_outputs')

    version = sys.argv[1]
    var = sys.argv[2]
    date = '062520'

    dataset_str = version + '_' + var + '_' + date + '.pkl'
    data = read_pickle(dataset_str)
    data = data.drop(data[data['dimensionality']==0].index) # remove rows with dimensionality zero

    peak_locations = []
    range_ys = []
    max_skills = []
    slopes = []
    labels = []
    obs_binary = []
    colors = []

    print('running for full dataset')
    avgs_x, pred_y = computil.plot_scatter_x_dimensionality_y(data, metric='hist_int', ystr='forecast', ylim=[0,0.8], var=var)
    p, r, m, s =  computil.compute_statistics_of_fit(avgs_x, pred_y)
    peak_locations.append(p)
    range_ys.append(r)
    max_skills.append(m)
    slopes.append(s)
    labels.append('All runs')
    obs_binary.append(0)

    data_obs_subset = data.loc[~(data.index.str.endswith('f'))]
    print('running for all obs only')
    avgs_x, pred_y = computil.plot_scatter_x_dimensionality_y(data_obs_subset, metric='hist_int', ystr='forecast', ylim=[0,0.8], subset='obs_only', var=var)
    p, r, m, s =  computil.compute_statistics_of_fit(avgs_x, pred_y)
    peak_locations.append(p)
    range_ys.append(r)
    max_skills.append(m)
    slopes.append(s)
    labels.append('All runs with obs')
    obs_binary.append(1)

    # <><><><><><><><><><><><><><><><><><><><>
    # <><><><> COMPLEXITY vs ACCURACY <><><><>
    # <><><><><><><><><><><><><><><><><><><><>

    for experiment in computil.get_experiments(type='numeric'):
        to_plot = computil.subset_df_by_substring(data, experiment)
        avgs_x, pred_y = computil.plot_scatter_x_dimensionality_y(data.loc[to_plot], metric='hist_int', ystr='forecast', ylim=[0,0.8], subset=experiment, var=var)
        p, r, m, s =  computil.compute_statistics_of_fit(avgs_x, pred_y)
        peak_locations.append(p)
        range_ys.append(r)
        max_skills.append(m)
        slopes.append(s)
        labels.append(experiment)
        obs_binary.append(0)

        to_plot = computil.subset_df_by_substring(data_obs_subset, experiment)
        avgs_x, pred_y = computil.plot_scatter_x_dimensionality_y(data_obs_subset.loc[to_plot], metric='hist_int', ystr='forecast', ylim=[0,0.8], subset=experiment + '_obs', var=var)
        p, r, m, s =  computil.compute_statistics_of_fit(avgs_x, pred_y)
        peak_locations.append(p)
        range_ys.append(r)
        max_skills.append(m)
        slopes.append(s)
        labels.append(experiment + '_obs')
        obs_binary.append(1)

    for experiment_letter in ['a','b','c','d','e','f']:
        to_plot = data[data.index.str.endswith(experiment_letter)]
        avgs_x, pred_y = computil.plot_scatter_x_dimensionality_y(to_plot, metric='hist_int', ystr='forecast', ylim=[0,0.8], subset='exp' + experiment_letter, var=var)
        p, r, m, s =  computil.compute_statistics_of_fit(avgs_x, pred_y)
        peak_locations.append(p)
        range_ys.append(r)
        max_skills.append(m)
        slopes.append(s)
        labels.append('exp' + experiment_letter)
        obs_binary.append(0)

        to_plot = data_obs_subset[data_obs_subset.index.str.endswith(experiment_letter)]
        avgs_x, pred_y = computil.plot_scatter_x_dimensionality_y(to_plot, metric='hist_int', ystr='forecast', ylim=[0,0.8], subset='exp' + experiment_letter + '_obs', var=var)
        p, r, m, s =  computil.compute_statistics_of_fit(avgs_x, pred_y)
        peak_locations.append(p)
        range_ys.append(r)
        max_skills.append(m)
        slopes.append(s)
        labels.append('exp' + experiment_letter + '_obs')
        obs_binary.append(1)

    for site in computil.site_years()['sites']:
        to_plot = computil.subset_df_by_substring(data, site)
        avgs_x, pred_y = computil.plot_scatter_x_dimensionality_y(data.loc[to_plot], metric='hist_int', ystr='forecast', ylim=[0,0.8], subset=site, var=var)
        p, r, m, s =  computil.compute_statistics_of_fit(avgs_x, pred_y)
        peak_locations.append(p)
        range_ys.append(r)
        max_skills.append(m)
        slopes.append(s)
        labels.append(site)
        obs_binary.append(0)

        to_plot = computil.subset_df_by_substring(data_obs_subset, site)
        avgs_x, pred_y = computil.plot_scatter_x_dimensionality_y(data_obs_subset.loc[to_plot], metric='hist_int', ystr='forecast', ylim=[0,0.8], subset=site + '_obs', var=var)
        p, r, m, s =  computil.compute_statistics_of_fit(avgs_x, pred_y)
        peak_locations.append(p)
        range_ys.append(r)
        max_skills.append(m)
        slopes.append(s)
        labels.append(site + '_obs')
        obs_binary.append(1)

    for EDC in ['_EDC', '_noEDC']:
        to_plot = computil.subset_df_by_substring(data, EDC)
        avgs_x, pred_y = computil.plot_scatter_x_dimensionality_y(data.loc[to_plot], metric='hist_int', ystr='forecast', ylim=[0,0.8], subset=EDC, var=var)
        p, r, m, s =  computil.compute_statistics_of_fit(avgs_x, pred_y)
        peak_locations.append(p)
        range_ys.append(r)
        max_skills.append(m)
        slopes.append(s)
        labels.append(EDC.replace('_',''))
        obs_binary.append(0)

        to_plot = computil.subset_df_by_substring(data_obs_subset, EDC)
        avgs_x, pred_y = computil.plot_scatter_x_dimensionality_y(data_obs_subset.loc[to_plot], metric='hist_int', ystr='forecast', ylim=[0,0.8], subset='obs' + EDC, var=var)
        p, r, m, s =  computil.compute_statistics_of_fit(avgs_x, pred_y)
        peak_locations.append(p)
        range_ys.append(r)
        max_skills.append(m)
        slopes.append(s)
        labels.append(EDC.replace('_','') + '_obs')
        obs_binary.append(1)

    computil.plot_statistics_of_fit_twoaxes(peak_locations, range_ys, labels, np.asarray(obs_binary))
    computil.plot_statistics_of_fit(peak_locations, range_ys, max_skills, slopes, labels, np.asarray(obs_binary))

    return

if __name__=='__main__':
    main()
