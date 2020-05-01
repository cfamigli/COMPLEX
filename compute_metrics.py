
import numpy as np
from pandas import DataFrame, to_pickle
import complex_utilities as computil
from histogram_intersection import histogram_intersection
import glob
import os
import sys
import warnings

#warnings.filterwarnings('ignore')

def main():
    version = 'v1.3'
    os.chdir('../data/COMPLEX_' + version)
    models = sorted([model for model in os.listdir(".")])
    models = [el for el in models if el[0]!='.']
    print(models)

    var = sys.argv[1]

    COMPLEX = DataFrame()
    for model in models:
        print(model)
        os.chdir(model)
        sites_EDCs = sorted([site_EDC for site_EDC in os.listdir(".")])
        sites_EDCs = [el for el in sites_EDCs if el[0]!='.']

        for site_EDC in sites_EDCs:
            print(site_EDC)

            # get observations and calibration indices
            site_obs_dir = '../../COMPLEX_input_files/' + site_EDC[:6]
            os.chdir(site_obs_dir)
            site_obs, site_obs_unc = computil.get_observations(var=var)
            site_calibration_indices = computil.calibration_period()
            os.chdir('../../COMPLEX_' + version + '/' + model + '/' + site_EDC)

            experiment_files = sorted(glob.glob('*' + var + '_*.csv'))

            for experiment_file in experiment_files:
                experiment = experiment_file[10:15]
                print(experiment_file)
                print(experiment)

                parameter_file = glob.glob('*' + experiment + '*parameters_*.csv')[0]
                parameter_data = computil.csv_to_np(parameter_file, header=None)
                dimensionality = computil.do_PCA(parameter_data, parameter_data.shape[1])

                site_pred = computil.csv_to_np(experiment_file, header=None)
                assert site_pred.shape[1] == site_obs.shape[0]

                cal_period_stop = site_calibration_indices[experiment].values[0]

                observational_error = np.nanmean(site_obs_unc) if len(np.unique(site_obs_unc))==1 else site_obs_unc

                computil.plot_time_series_with_spread(site_obs, site_pred,
                    observational_error, cal_period_stop, var=var,
                    title=model + '_' + site_EDC + '_' + experiment + '_' + var)

                hist_int_calibration, hist_int_forecast = histogram_intersection(site_obs,
                    site_pred, cal_period_stop, observational_error=observational_error)

                print(hist_int_calibration, hist_int_forecast)

                R2_calibration, RMSE_calibration = computil.calc_R2_RMSE(site_obs[:cal_period_stop],
                    np.nanmedian(site_pred[:,:cal_period_stop], axis=0))

                R2_forecast, RMSE_forecast = computil.calc_R2_RMSE(site_obs[cal_period_stop:],
                    np.nanmedian(site_pred[:,cal_period_stop:], axis=0))

                good_flag = 0 if np.nanmedian(site_pred[:,0], axis=0)>20 else 1
                # good_flag is 1 if run is "good" (no sharp burn-in)

                nee_bool = 1. if (experiment[-1]=='a') | (experiment[-1]=='b') | (experiment[-1]=='c') else 0.

                COMPLEX = computil.append_to_df(COMPLEX, model, site_EDC, experiment, good_flag,
                    dimensionality, hist_int_calibration, hist_int_forecast, R2_calibration,
                    R2_forecast, RMSE_calibration, RMSE_forecast, nee_bool)

            os.chdir('..')

        os.chdir('..')

    os.chdir('..')
    COMPLEX = COMPLEX.T
    COMPLEX.to_pickle('analysis_outputs/' + version + '_' + var + '_042720.pkl')
    return

if __name__=='__main__':
    main()
