import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the directory containing CSV files
directory = 'all_ind_aggrzcp_res'

# Read and combine all CSV files
all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
data_list = [pd.read_csv(file, header=None) for file in all_files]

# Combine all data into a single DataFrame
combined_data = pd.concat(data_list, ignore_index=True)

# Create a new directory for saving the graphs
output_directory = 'ablation_30b'
os.makedirs(output_directory, exist_ok=True)

# Set column names based on the provided data format
combined_data.columns = [
    'prune_mode', 'col1', 'col2', 'col3', 'zcp', 'col5', 'col6', 'col7', 'col8', 
    'col9', 'sparsity', 'predmethod', 'col11', 'task', 'col13', 'col14', 'perplexity', 
    'col16', 'col17'
]

# Extract unique values for zcp, prune_mode, and predmethod
zcps = combined_data['zcp'].unique()
prune_modes = combined_data['prune_mode'].unique()
predmethods = combined_data['predmethod'].unique()

# Plotting
for prune_mode in prune_modes:
    plt.figure()
    for zcp in zcps:
        for predmethod in predmethods:
            subset = combined_data[
                (combined_data['zcp'] == zcp) & 
                (combined_data['prune_mode'] == prune_mode) & 
                (combined_data['predmethod'] == predmethod)
            ]
            if not subset.empty:
                plt.plot(subset['sparsity'], subset['perplexity'], marker='o', label=f'{zcp}_{predmethod}')
    
    plt.title(f'Prune Mode: {prune_mode}')
    plt.xlabel('Sparsity')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_filename = f'{output_directory}/plot_prunemode_{prune_mode}.png'
    plt.savefig(plot_filename)
    plt.close()
