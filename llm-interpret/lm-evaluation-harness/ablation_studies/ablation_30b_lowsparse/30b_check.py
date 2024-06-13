import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the directory containing CSV files
directory = 'all_ind_aggrzcp_res_opt30b_main'

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

zcpmap = {"plainact": "PlainAct", "l2_norm": "L2Norm"}
predmap = {"predictorL": "ShadowLLM", "dejavu": "DejaVu-Style"}
pmode_map = {"global": "Global", "perlayer": "Local"}

# Set plot styles for IEEE scientific standard
plt.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.6,
    'grid.linestyle': '--'
})

fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

for i, prune_mode in enumerate(prune_modes):
    ax = axs[i]
    for zcp in zcps:
        for predmethod in predmethods:
            subset = combined_data[
                (combined_data['zcp'] == zcp) & 
                (combined_data['prune_mode'] == prune_mode) & 
                (combined_data['predmethod'] == predmethod)
            ]
            if not subset.empty:
                if f'{zcp}_{predmethod}' in ["plainact_predictorL", "l2_norm_dejavu"]:
                    subset = subset.sort_values(by='sparsity')
                    subset = subset[subset['sparsity'] <= 50]
                    ax.plot(subset['sparsity'], subset['perplexity'], marker='o', label=f'{predmap[predmethod]} ({zcpmap[zcp]})')

    
    ax.set_title(f'{pmode_map[prune_mode]} Pruning On OPT-30B', fontsize=24)
    ax.set_xlabel('Sparsity (%)', fontsize=24)
    ax.set_ylabel('Perplexity', fontsize=24)
    ax.legend(fontsize=22, loc='upper left', ncol=1)
    if i == 0:
        # set ylim as 40
        ax.set_ylim(10, 40)
    else:
        ax.set_ylim(10, 15)

    ax.grid(True)
plt.tight_layout()
output_path = os.path.join(output_directory, 'stacked_perplexity_plots.pdf')
plt.savefig(output_path, bbox_inches='tight')
plt.close()
