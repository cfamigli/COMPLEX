
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
    date = '052120'

    dataset_str = version + '_' + var + '_' + date + '.pkl'
    data = read_pickle(dataset_str)

    #data = v1_3.loc[~np.isnan(v1_3['dimensionality'])]
    print('running for full dataset')
    computil.run_plots(data, var=var)

    data_nee_subset = data.loc[data['nee']==1]
    print('running for nee only')
    computil.run_plots(data_nee_subset, subset_str='nee_only', var=var)

    data_no_nee_subset = data.loc[(data['nee']==0) & ~(data.index.str.endswith('f'))]
    print('running for other obs only')
    computil.run_plots(data_no_nee_subset, subset_str='obs_no_nee_only', var=var)

    model_list = ['C1','C2','C3','C4','C5','C6','C7','C8','E1','G1','G2','G3','G4','S1','S2','S4']

    # <><><><><><><><><><><><><><><><><><><><>
    # <><><><> COMPLEXITY vs ACCURACY <><><><>
    # <><><><><><><><><><><><><><><><><><><><>

    model_pars = computil.raw_complexity()
    average_metric = []
    std_metric = []
    absolute_complexity = []
    for model in model_list:
        to_plot = computil.subset_df_by_substring(data, model)
        average_metric.append(data.loc[to_plot]['hist_int_forecast'].mean())
        std_metric.append(data.loc[to_plot]['hist_int_forecast'].std())
        absolute_complexity.append(model_pars.loc[model_pars['models']==model]['npars'].values)
        computil.run_plots(data.loc[to_plot], subset_str=model, var=var)

    computil.plot_scatter_model_avg_x_dimensionality_y(lx=absolute_complexity,
        ly=average_metric, ey=std_metric, var=var)

    for experiment in computil.get_experiments():
        to_plot = computil.subset_df_by_substring(data, experiment)
        computil.run_plots(data.loc[to_plot], subset_str=experiment, var=var)

    for experiment_letter in ['a','b','c','d','e','f']:
        to_plot = data[data.index.str.endswith(experiment_letter)]
        computil.run_plots(to_plot, subset_str='exp' + experiment_letter, var=var)

    for site in computil.site_years()['sites']:
        to_plot = computil.subset_df_by_substring(data, site)
        computil.run_plots(data.loc[to_plot], subset_str=site, var=var)

    for site in computil.site_years()['sites']:
        sites = computil.subset_df_by_substring(data, site)
        for EDC in ['_EDC', '_noEDC']:
            to_plot = computil.subset_list_by_substring(sites, EDC)
            computil.run_plots(data.loc[to_plot], subset_str=site + EDC, var=var)

    for model in model_list:
        models = computil.subset_df_by_substring(data, model)
        for EDC in ['_EDC', '_noEDC']:
            to_plot = computil.subset_list_by_substring(models, EDC)
            computil.run_plots(data.loc[to_plot], subset_str=model + EDC, var=var)

    for model in model_list:
        models = computil.subset_df_by_substring(data, model)
        for site in computil.site_years()['sites']:
            to_plot = computil.subset_list_by_substring(models, site)
            computil.run_plots(data.loc[to_plot], subset_str=model + '_' + site, var=var)

    for EDC in ['_EDC', '_noEDC']:
        to_plot = computil.subset_df_by_substring(data, EDC)
        computil.run_plots(data.loc[to_plot], subset_str=EDC, var=var)

        to_plot = computil.subset_df_by_substring(data_nee_subset, EDC)
        computil.run_plots(data_nee_subset.loc[to_plot], subset_str='nee' + EDC, var=var)

    for EDC in ['_EDC', '_noEDC']:
        to_plot = computil.subset_df_by_substring(data, EDC)
        data_to_plot = data.loc[to_plot]
        for experiment_letter in ['a','b','c','d','e','f']:
            to_plot = data_to_plot[data_to_plot.index.str.endswith(experiment_letter)]
            computil.run_plots(to_plot, subset_str='exp' + experiment_letter + EDC, var=var)

    for experiment in computil.get_experiments():
        experiments = computil.subset_df_by_substring(data, experiment)
        for model in model_list:
            to_plot = computil.subset_list_by_substring(experiments, model)
            computil.run_plots(data.loc[to_plot], subset_str=model + '_' + experiment, var=var)

    # <><><><><><><><><><><><><><><><><><><>
    # <><><><> PROCESS vs ACCURACY <><><><>
    # <><><><><><><><><><><><><><><><><><><>

    for process in data.columns[9:]:
        computil.plot_scatter_x_process_y_skill(data, process=process, ystr='forecast', metric='hist_int', var=var)
        computil.plot_scatter_x_process_y_skill(data.loc[computil.subset_df_by_substring(data, '_EDC')],
            process=process, ystr='forecast', metric='hist_int', subset='EDCs', var=var)

    # <><><><><><><><><><><><><><><><><><><>
    # <><><><> ACCURACY vs ACCURACY <><><><>
    # <><><><><><><><><><><><><><><><><><><>

    for [xstr, ystr] in [['calibration', 'forecast'],['forecast', 'diff']]:
        for metric in ['hist_int']:
            '''computil.plot_scatter_x_performance_y_multicolor(data,
                model_list, xstr=xstr, ystr=ystr, subset='models', var=var)

            computil.plot_scatter_x_performance_y_multicolor(data,
                computil.get_experiments(type='numeric'),
                xstr=xstr, ystr=ystr, subset='experiments', var=var)

            computil.plot_scatter_x_performance_y_multicolor(data,
                computil.site_years()['sites'],
                xstr=xstr, ystr=ystr, subset='sites', var=var)

            computil.plot_scatter_x_performance_y_multicolor(data,
                ['_EDC', '_noEDC'],
                xstr=xstr, ystr=ystr, subset='EDCs', var=var)'''

            computil.plot_scatter_x_performance_y_dimensionality(data,
                ['_EDC', '_noEDC'], xstr=xstr, ystr=ystr, metric=metric, subset='EDCs', var=var)

            computil.plot_scatter_x_performance_y_dimensionality(data,
                ['nee', 'no_nee'], xstr=xstr, ystr=ystr, metric=metric, subset='nee', var=var)

            computil.plot_scatter_x_performance_y_dimensionality(data,
                ['nee_EDC', 'no_nee_noEDC'], xstr=xstr, ystr=ystr, metric=metric, subset='nee_EDC', var=var)

            computil.plot_scatter_x_performance_y_dimensionality(data,
                model_list, xstr=xstr, ystr=ystr, metric=metric, subset='models', var=var)

            computil.plot_scatter_x_performance_y_dimensionality(data,
                computil.site_years()['sites'],
                xstr=xstr, ystr=ystr, metric=metric, subset='sites', var=var)

            computil.plot_scatter_x_performance_y_dimensionality(data,
                computil.get_experiments(type='numeric'),
                xstr=xstr, ystr=ystr, metric=metric, subset='experiments', var=var)

            computil.plot_scatter_x_performance_y_dimensionality(data,
                ['a','b','c','d','e','f'],
                xstr=xstr, ystr=ystr, metric=metric, subset='experiment_letters', var=var)


            # <><><><><><><><><><><><><><><><><><><>
            # <><><><> EDC ONLY <><><><>
            # <><><><><><><><><><><><><><><><><><><>

            to_plot = computil.subset_df_by_substring(data, '_EDC')
            '''computil.plot_scatter_x_performance_y_multicolor(data.loc[to_plot],
                model_list, xstr=xstr, ystr=ystr, subset='models_EDC', var=var)

            computil.plot_scatter_x_performance_y_multicolor(data.loc[to_plot],
                computil.get_experiments(type='numeric'),
                xstr=xstr, ystr=ystr, subset='experiments_EDC', var=var)

            computil.plot_scatter_x_performance_y_multicolor(data.loc[to_plot],
                computil.site_years()['sites'],
                xstr=xstr, ystr=ystr, subset='sites_EDC', var=var)

            computil.plot_scatter_x_performance_y_multicolor(data.loc[to_plot],
                ['_EDC', '_noEDC'],
                xstr=xstr, ystr=ystr, subset='EDCs_EDC', var=var)'''

            computil.plot_scatter_x_performance_y_dimensionality(data.loc[to_plot],
                model_list, xstr=xstr, ystr=ystr, metric=metric, subset='models_EDC', var=var)

            computil.plot_scatter_x_performance_y_dimensionality(data.loc[to_plot],
                computil.get_experiments(type='numeric'),
                xstr=xstr, ystr=ystr, metric=metric, subset='experiments_EDC', var=var)

            computil.plot_scatter_x_performance_y_dimensionality(data.loc[to_plot],
                computil.site_years()['sites'],
                xstr=xstr, ystr=ystr, metric=metric, subset='sites_EDC', var=var)

            computil.plot_scatter_x_performance_y_dimensionality(data.loc[to_plot],
                ['a','b','c','d','e','f'],
                xstr=xstr, ystr=ystr, metric=metric, subset='experiment_letters_EDC', var=var)


            # <><><><><><><><><><><><><><><><><><><>
            # <><><><> NEE ONLY <><><><>
            # <><><><><><><><><><><><><><><><><><><>

            to_plot = data[data['nee']==1]
            computil.plot_scatter_x_performance_y_dimensionality(to_plot,
                model_list, xstr=xstr, ystr=ystr, metric=metric, subset='models_nee', var=var)

            computil.plot_scatter_x_performance_y_dimensionality(to_plot,
                computil.get_experiments(type='numeric'),
                xstr=xstr, ystr=ystr, metric=metric, subset='experiments_nee', var=var)

            computil.plot_scatter_x_performance_y_dimensionality(to_plot,
                computil.site_years()['sites'],
                xstr=xstr, ystr=ystr, metric=metric, subset='sites_nee', var=var)

    return

if __name__=='__main__':
    main()
