
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

    for model in models:

        exp_count = 0
        one_exp_count = 0
        two_exp_count = 0
        any_exp_count = 0

        print(model)
        os.chdir(model)
        sites_EDCs = sorted([site_EDC for site_EDC in os.listdir(".")])
        sites_EDCs = [el for el in sites_EDCs if el[0]!='.']

        for site_EDC in sites_EDCs:
            os.chdir(site_EDC)

            experiment_files = sorted(glob.glob('*' + var + '_*.csv'))

            for experiment_file in experiment_files:
                experiment = experiment_file[10:15]

                parameter_file = glob.glob('*' + experiment + '*parameters_*.csv')[0]
                parameter_data = computil.csv_to_np(parameter_file, header=None)

                n_unique = len(np.unique(parameter_data[:,0]))

                if (n_unique==1):#
                    one_exp_count += 1
                elif (n_unique==2):
                    two_exp_count += 1
                elif (n_unique!=parameter_data.shape[0]):
                    any_exp_count += 1


                exp_count += 1

            os.chdir('..')

        os.chdir('..')
        print(model)
        print(exp_count)
        print(one_exp_count)
        print(two_exp_count)
        print(any_exp_count)


    os.chdir('..')
    return

if __name__=='__main__':
    main()
