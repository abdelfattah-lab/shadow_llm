import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Set plot styles for IEEE scientific standard
plt.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.6,
    'grid.linestyle': '--'
})

# Path to the directory containing CSV files
directory = 'all_ind_aggrzcp_res'
output_directory = 'ablation_pred_strategy'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read and combine all CSV files
all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
data_list = [pd.read_csv(file, header=None) for file in all_files]
data = pd.concat(data_list, ignore_index=True)

# Save data as a csv to output_directory
data.to_csv(os.path.join(output_directory, 'all_data.csv'))

# Extract relevant columns
data = data.iloc[:, [0, 1, 4, 11, 16]]
data.columns = ['pruning_strategy', 'sparsity', 'proxy', 'strategy', 'perplexity']
# if strategy == 'original', remove it
data = data[data['strategy'] != 'original']
# # only keep proxy == "plaianct"
# data = data[data['proxy'] == 'plainact']
# for the dejavu, keep only l2_norm
# data2 = data[(data['proxy'] == 'l2_norm') & (data['strategy'] == 'dejavu')]
# data1 = data[(data['proxy'] == 'fisher') & (data['strategy'] == 'predictorL')]
# data = pd.concat([data1, data2])

# Update strategy names
data['strategy'] = data['strategy'].replace({'predictorL': 'ShadowLLM', 'dejavu': 'DejaVu'})

# # keep proxy == "plainact" for "predictorL"
# data1 = data[(data['proxy'] == 'fisher') & (data['strategy'] == 'ShadowLLM')]
# # only keep proxy == "l2_norm" for "dejavu"
# data = data[(data['proxy'] == 'l2_norm') & (data['strategy'] == 'DejaVu')]
# # Combine the two dataframes
# data = pd.concat([data1, data])
# Take the average across all proxies
data_grouped = data.groupby(['sparsity', 'pruning_strategy', 'strategy']).mean().reset_index()

# Sort by sparsity for proper line plotting
data_grouped = data_grouped.sort_values(by=['sparsity'])

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

line_styles = {'global': '--', 'perlayer': '-'}

for (pruning_strategy, strategy), grp in data_grouped.groupby(['pruning_strategy', 'strategy']):
    psc = "Global" if pruning_strategy == "global" else "Per-layer"
    label = f'{strategy} ({psc})'
    if "predictor" in label:
        continue
    line_style = line_styles[pruning_strategy] if pruning_strategy in line_styles else '-'
    ax = grp.plot(ax=ax, kind='line', x='sparsity', y='perplexity', label=label, marker='o', linestyle=line_style)

plt.title('Perplexity vs Sparsity on WikiText2 For OPT-1.3B')
# Y Axis label accuracy, 
# X axis label "Sparsity"
plt.ylabel('Perplexity')
plt.xlabel('Sparsity (%)')
# Set legend fontsize as 18
# plt.legend(fontsize=18)
# Add grid

# Reorder legend
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(l) for l in sorted(labels, key=lambda x: ('DejaVu' in x, x), reverse=True)]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=18)

plt.grid(True)
plt.tight_layout()
# Save the plot
plt.savefig(os.path.join(output_directory, 'perplexity_vs_sparsity.pdf'))

