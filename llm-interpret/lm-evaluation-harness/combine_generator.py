import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# model_dirs = ['opt-1.3b', 'opt-6.7b', 'opt-13b']

# for msize in model_dirs:
#     dir_to_read = f'zcps/{msize}'
#     # Read every pkl file that doesnt contain trace in the name using os.listdir
#     all_files = [os.path.join(dir_to_read, f) for f in os.listdir(dir_to_read) if f.endswith('.pkl') and 'trace' not in f]
#     # convert all_files to a dictionary of lists, with fname.split("_")[0] as key
#     data_dict = {}
#     for file in all_files:
#         fname = file.split("/")[-1]
#         key = fname.split("_")[0]
#         if key not in data_dict:
#             data_dict[key] = []
#         data_dict[key].append(pd.read_pickle(file))
    
#     # for each key in data_dict, do np.asarray(data_dict[key]) and np.mean(axis=0) to get the mean of the list of arrays
#     for key in data_dict:
#         data_dict[key] = np.mean(np.asarray(data_dict[key]), axis=0)
#     # save it back as a pickle file with name f'{key}_all_0.pkl'
#     for key in data_dict:
#         pd.to_pickle(data_dict[key], f'{dir_to_read}/{key}_all_0.pkl')


model_dirs = ['opt-1.3b', 'opt-6.7b', 'opt-13b']

def combine_dicts(ldict):
    combined_dict = {}
    offset = 0
    for d in ldict:
        for k, v in d.items():
            combined_dict[offset + k] = v
        offset += len(d)
    return combined_dict


for msize in model_dirs:
    dir_to_read = f'zcps/{msize}'
    # Read every pkl file that doesnt contain trace in the name using os.listdir
    all_files = [os.path.join(dir_to_read, f) for f in os.listdir(dir_to_read) if f.endswith('.pkl') and 'trace' in f]
    # convert all_files to a dictionary of lists, with fname.split("_")[0] as key
    data_dict = {}
    for file in all_files:
        fname = file.split("/")[-1]
        key = fname.split("_")[0]
        if key not in data_dict:
            data_dict[key] = []
        data_dict[key].append(pd.read_pickle(file))
    for key in data_dict:
        data_dict[key] = combine_dicts(data_dict[key])
    # # for each key in data_dict, do np.asarray(data_dict[key]) and np.mean(axis=0) to get the mean of the list of arrays
    # # save it back as a pickle file with name f'{key}_all_0.pkl'
    for key in data_dict:
        pd.to_pickle(data_dict[key], f'{dir_to_read}/{key}_trace_all_0.pkl')
