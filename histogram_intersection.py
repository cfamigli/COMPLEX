
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def histogram_intersection(observational_data, cardamom_data,
    calibration_period_stop=None, observational_error=0.5):
    n_steps = len(observational_data)
    n_ensembles = cardamom_data.shape[0]

    hist_int = np.ones(n_steps)*np.nan
    for step in range(n_steps):
        if observational_data[step]==-9999:
            continue
        else:
            if np.ndim(observational_error)==0:
                obs_dist = np.random.normal(observational_data[step], observational_error,
                    n_ensembles)
            else:
                obs_dist = np.random.normal(observational_data[step], observational_error[step],
                    n_ensembles)

            cardamom_dist = cardamom_data[:,step]
            cardamom_dist[cardamom_dist==-9999] = float('nan')
            cardamom_dist = cardamom_dist[np.isfinite(cardamom_dist)]

            if len(cardamom_dist)==0:
                continue

            else:
                rng = [np.nanmin([np.nanmin(obs_dist), np.nanmin(cardamom_dist)])-0.5, np.nanmax([np.nanmax(obs_dist), np.nanmax(cardamom_dist)])+0.5]

                p_obs, bins_obs, _ = plt.hist(obs_dist, range=rng, bins=50, density=True)
                p_cardamom, bins_cardamom, _ = plt.hist(cardamom_dist, range=rng, bins=50, density=True)

                plt.close()

                min_overlap = np.minimum(p_obs, p_cardamom)
                bin_width = abs(bins_obs[0]-bins_obs[1])
                overlap_area = np.sum(bin_width*min_overlap)
                total_model_area = np.sum(bin_width*p_cardamom)
                hist_int[step] = overlap_area / total_model_area

    average_hist_int_calibration = np.nanmean(hist_int[:calibration_period_stop])
    average_hist_int_forecast = np.nanmean(hist_int[calibration_period_stop:])
    return (average_hist_int_calibration, average_hist_int_forecast)
