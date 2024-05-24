import pandas as pd
import matplotlib.pyplot as plt
import os

import numpy as np
# Path to the directory containing CSV files
directory = 'ind_aggrzcp_res'

# Read and combine all CSV files
all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
data_list = [pd.read_csv(file, header=None) for file in all_files]
data = pd.concat(data_list, ignore_index=True)

# # Process the first column to extract metric names
# data[0] = data[0].str.rsplit('_', n=1).str[0]

# data[0] = data[0].apply(lambda x: x.rsplit('_', n=1)[0] if 'winogrande' not in x else x.rsplit('_', n=2)[0])
data[0] = data[0].apply(lambda x: x.rsplit('_', 2)[0] if 'winogrande' in x else x.rsplit('_', 1)[0])

# Drop any unnecessary columns beyond the fourth
data = data.iloc[:, :5]
# Rename columns for clarity
data.columns = ['metric_name', 'sparsity', 'dataset', 'method', 'accuracy']

data['metric_name'] = data['metric_name'].str.replace(r'^oracle.*', 'oracle', regex=True)
data = data[~data['dataset'].isin(['wic', 'wsc', 'rte', 'winogrande'])]

# Calculate geometric mean across datasets for each metric and sparsity
geo_mean = data.groupby(['metric_name', 'sparsity']).agg({'accuracy': lambda x: np.exp(np.mean(np.log(x)))}).reset_index()
geo_mean['dataset'] = 'Geomean'

# Append the geometric mean as a new dataset
data = pd.concat([data, geo_mean], ignore_index=True)


# make directory with name 'aggr_zcp_perf' if it doesn't exist
if not os.path.exists('aggr_zcp_perf'):
    os.makedirs('aggr_zcp_perf')
# Plotting
for dataset in data['dataset'].unique():
    plt.figure()
    dataset_data = data[data['dataset'] == dataset]
    # Sort dataset_data by sparsity
    dataset_data = dataset_data.sort_values('sparsity')
    for metric in dataset_data['metric_name'].unique():
        metric_data = dataset_data[dataset_data['metric_name'] == metric]
        plt.plot(metric_data['sparsity'], metric_data['accuracy'], marker='o', linestyle='-', label=metric)
    # make x axis log power of 2
    # plt.xscale('log', basex=2)
    plt.xlabel('Sparsity')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Sparsity for {dataset}')
    plt.legend()
    plt.savefig(f'./aggr_zcp_perf/{dataset}_aggr_zcp.png')
    plt.close()