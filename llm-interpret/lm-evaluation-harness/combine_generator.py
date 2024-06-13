import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

model_dirs = ['opt-1.3b']


fs_set = 0

for msize in model_dirs:
    dir_to_read = f'zcps/{msize}'
    # Read every pkl file that doesnt contain trace in the name using os.listdir
    all_files = [os.path.join(dir_to_read, f) for f in os.listdir(dir_to_read) if f.endswith('.pkl') and 'trace' not in f and 'fc1' not in f and f'_{fs_set}' in f]
    # convert all_files to a dictionary of lists, with fname.split("_")[0] as key
    data_dict = {}
    for file in tqdm(all_files):
        fname = file.split("/")[-1]
        key = fname.split("_")[0]
        if key == "l2":
            key = "l2_norm"
        if key == "grad":
            key = "grad_norm"
        if key not in data_dict:
            data_dict[key] = []
        samp_data = pd.read_pickle(file)
        # convert to numpy and 0-1 normalize
        samp_data = np.asarray(samp_data)
        data_dict[key].append(samp_data)
    
    # for each key in data_dict, do np.asarray(data_dict[key]) and np.mean(axis=0) to get the mean of the list of arrays
    for key in tqdm(data_dict):
        try:
            data_dict[key] = np.mean(np.asarray(data_dict[key]), axis=0)
        except:
            import pdb; pdb.set_trace()
    # # save it back as a pickle file with name f'{key}_all_0.pkl'
    for key in tqdm(data_dict):
        pd.to_pickle(data_dict[key], f'{dir_to_read}/{key}_all_{fs_set}.pkl')

for msize in model_dirs:
    dir_to_read = f'zcps/{msize}'
    # Read every pkl file that doesnt contain trace in the name using os.listdir
    all_files = [os.path.join(dir_to_read, f) for f in os.listdir(dir_to_read) if f.endswith('.pkl') and 'trace' not in f and 'fc1' in f and f'_{fs_set}' in f]
    # convert all_files to a dictionary of lists, with fname.split("_")[0] as key
    data_dict = {}
    for file in tqdm(all_files):
        fname = file.split("/")[-1]
        key = fname.split("_")[0]
        if key == "l2":
            key = "l2_norm"
        if key == "grad":
            key = "grad_norm"
        if key not in data_dict:
            data_dict[key] = []
        samp_data = pd.read_pickle(file)
        # convert to numpy and 0-1 normalize
        samp_data = np.asarray(samp_data)
        data_dict[key].append(samp_data)
    
    # for each key in data_dict, do np.asarray(data_dict[key]) and np.mean(axis=0) to get the mean of the list of arrays
    for key in tqdm(data_dict):
        try:
            data_dict[key] = np.mean(np.asarray(data_dict[key]), axis=0)
        except:
            import pdb; pdb.set_trace()
    # # save it back as a pickle file with name f'{key}_all_0.pkl'
    for key in tqdm(data_dict):
        pd.to_pickle(data_dict[key], f'{dir_to_read}/{key}_fc1_all_{fs_set}.pkl')




import torch

# make temp directory if it doesnt exist
for msize in model_dirs:
    os.makedirs(f'temp_fs_{msize}', exist_ok=True)
    dir_to_read = f'zcps/{msize}'
    # Read every pkl file that doesnt contain trace in the name using os.listdir
    all_files = [os.path.join(dir_to_read, f) for f in os.listdir(dir_to_read) if f.endswith('.pkl') and 'trace' in f and f'_{fs_set}' in f]
    # convert all_files to a dictionary of lists, with fname.split("_")[0] as key
    for file in tqdm(all_files):
        fname = file.split("/")[-1]
        key = fname.split("_")[0]
        trace_read = pd.read_pickle(file)
        for tidx in range(len(trace_read)):
            trace_read[tidx] = list(trace_read[tidx])
            trace_read[tidx][1] = [x.cpu() for x in trace_read[tidx][1]]
        torch.cuda.empty_cache()
        pd.to_pickle(trace_read, f'temp_fs_{msize}/{fname}.pkl')


def combine_dicts(ldict):
    combined_dict = {}
    offset = 0
    for d in ldict:
        for k, v in d.items():
            combined_dict[offset + k] = v
        offset += len(d)
    return combined_dict

# Now, read the saved files, combine them and save them back
for msize in model_dirs:
    dir_to_read = f'temp_fs_{msize}'
    # Read every pkl file that doesnt contain trace in the name using os.listdir
    all_files = [os.path.join(dir_to_read, f) for f in os.listdir(dir_to_read) if f.endswith('.pkl') and f'_{fs_set}' in f]
    # convert all_files to a dictionary of lists, with fname.split("_")[0] as key
    data_dict = {}
    for file in tqdm(all_files):
        fname = file.split("/")[-1]
        key = fname.split("_")[0]
        if key == "l2":
            key = "l2_norm"
        if key == "grad":
            key = "grad_norm"
        if key not in data_dict:
            data_dict[key] = []
        trace_read = pd.read_pickle(file)
        data_dict[key].append(trace_read)

    for key in tqdm(data_dict):
        data_dict[key] = combine_dicts(data_dict[key])
    for key in tqdm(data_dict):
        pd.to_pickle(data_dict[key], f'zcps/{msize}/{key}_trace_all_{fs_set}.pkl')


