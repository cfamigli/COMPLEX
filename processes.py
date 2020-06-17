
import numpy as np
import complex_utilities as computil
import os
import sys
from pandas import read_pickle
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def main():
    os.chdir('../data/analysis_outputs')

    version = sys.argv[1]
    var = sys.argv[2]
    date = '052320'

    dataset_str = version + '_' + var + '_' + date + '.pkl'
    data = read_pickle(dataset_str)

    # <><><><><><><><><><><><><><><><><><><>
    # <><><><> PROCESS vs ACCURACY <><><><>
    # <><><><><><><><><><><><><><><><><><><>

    max_diff = []
    for process in data.columns[9:]:
        medians = computil.plot_scatter_x_process_y_skill(data, process=process, ystr='forecast', metric='hist_int', var=var)
        max_diff.append(max(medians) - min(medians))

        medians = computil.plot_scatter_x_process_y_skill(data.loc[computil.subset_df_by_substring(data, '_EDC')],
            process=process, ystr='forecast', metric='hist_int', subset='EDCs', var=var)

    computil.plot_scatter_x_maxdiff_y_process(max_diff, data.columns[9:], ystr='forecast', metric='hist_int', subset='', var='NEE')

    return

if __name__=='__main__':
    main()
