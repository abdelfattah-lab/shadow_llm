import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Path to the directory containing CSV files
directory = 'all_ind_aggrzcp_res'

"""
We wish to have sparsity on x axis (2nd column)
Accuracy on Y axis (last column)
combination of method and predictor type as the legend (1st column + "_" + 12th column)
"""



# Read and combine all CSV files
all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
data_list = [pd.read_csv(file, header=None) for file in all_files]
data = pd.concat(data_list, ignore_index=True)

# Drop any unnecessary columns beyond the fourth
data = data.iloc[:, :5]
# Rename columns for clarity
data.columns = ['metric_name', 'sparsity', 'dataset', 'method', 'accuracy']
data['metric_name'] = data['metric_name'].str.replace(r'^oracle.*', 'oracle', regex=True)

# Split 'metric_name' and extract the dataset name
data['dataset'] = data['metric_name'].apply(lambda x: x.rsplit('_', 2)[1] if 'winogrande' in x else x.rsplit('_', 1)[1])
data['metric_name'] = data['metric_name'].apply(lambda x: x.rsplit('_', 2)[0] if 'winogrande' in x else x.rsplit('_', 1)[0])

# Filter data to include only those with 'dejavu' or 'predictor' methods
data = data[data['method'].isin(['dejavu', 'predictor'])]

# Remove any entries missing in either type of method
filtered_data = data.groupby(['metric_name', 'sparsity', 'dataset']).filter(lambda x: len(x['method'].unique()) == 2)

# Pivot data to have 'dejavu' and 'predictor' in separate columns
pivot_data = filtered_data.pivot_table(index=['metric_name', 'sparsity', 'dataset'], columns='method', values='accuracy').reset_index()

# Calculate the accuracy difference
pivot_data['accuracy_diff'] = 100 * (pivot_data['predictor'] - pivot_data['dejavu']) / pivot_data['dejavu']

# Calculate mean accuracy differences across datasets for each metric and sparsity
mean_diff = pivot_data.groupby(['metric_name', 'sparsity']).agg({'accuracy_diff': 'mean'}).reset_index()

# Plotting
output_directory = 'all_zcp_aggr_vs_pred'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for metric in mean_diff['metric_name'].unique():
    plt.figure()
    metric_data = mean_diff[mean_diff['metric_name'] == metric]
    metric_data = metric_data.sort_values('sparsity')
    plt.plot(metric_data['sparsity'], metric_data['accuracy_diff'], marker='o', linestyle='-', label=metric)
    plt.xlabel('Sparsity')
    plt.ylabel('Accuracy Difference (Predictor - Dejavu)')
    plt.title(f'Accuracy Difference vs Sparsity for {metric}')
    plt.legend()
    plt.savefig(f'./{output_directory}/{metric}_aggr_zcp_diff.png')
    plt.close()
