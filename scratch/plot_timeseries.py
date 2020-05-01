
import numpy as np
import os
import glob
from pandas import read_csv
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def plot_time_series_with_spread(obs_data, pred_data, obs_unc, cal_period_stop=60, var='NEE', title=None):
    # this function requires:
    # obs_data: a vector of observations of length n timesteps
    # pred_data: a matrix of cardamom ensembles of size (m, n) = (ensembles, timesteps)
    # obs_unc: observational uncertainty, either one value (NEE) or time-varying (LAI)
    # cal_period_stop: a scalar value denoting the index between calibration and forecast. e.g., 36.
    # var: which variable we're plotting. default is 'NEE'
    # title: for plotting purposes only.

    o = np.copy(obs_data)
    p = np.copy(pred_data)
    o[obs_data==-9999] = float('nan') # remove fill values
    p[pred_data==-9999] = float('nan')

    if np.ndim(obs_unc)>0: # if observational uncertainty is a vector instead of a scalar
        obs_unc[obs_unc==-9999]=float('nan') # make sure it doesn't have any fill values

    plt.figure(figsize=(11,4))

    # shade forecast period
    plt.axvspan(cal_period_stop, p.shape[1], alpha=0.4, color='lightgray',
        label='forecast window', zorder=0)

    # plot cardamom ensembles and shade 5th to 95th percentile
    plt.fill_between(np.arange(p.shape[1]), np.nanpercentile(p, 95, axis=0),
        np.nanpercentile(p, 5, axis=0), facecolor='lightsteelblue', alpha=0.8, label='ensemble spread (predicted)', zorder=1)

    # plot cardamom ensemble mean
    plt.plot(np.nanmedian(p, axis=0), c='darkblue', linewidth=2.5,
        label='ensemble median (predicted)', zorder=2)

    # plot observations
    plt.plot(o, c='crimson', linewidth=2, label='observed', zorder=4)

    # plot observational uncertainty shading
    plt.fill_between(np.arange(p.shape[1]), o+obs_unc, o-obs_unc,
        facecolor='lightcoral', alpha=0.8, label='observational uncertainty', zorder=1.5)
        
    plt.ylabel(var)
    plt.xlabel('months after start')
    plt.legend(loc='best')
    plt.title(title)
    plt.show()
    plt.close()
    return
