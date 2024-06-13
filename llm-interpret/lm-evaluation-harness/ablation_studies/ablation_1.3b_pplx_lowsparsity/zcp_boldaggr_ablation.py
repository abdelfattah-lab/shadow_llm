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
    'grid.linestyle': '--',
})

# Path to the directory containing CSV files
directory = 'all_ind_aggrzcp_res'

# Read and combine all CSV files
all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
data_list = [pd.read_csv(file, header=None) for file in all_files]
data = pd.concat(data_list, ignore_index=True)
# make a directory called 'zcp_aggr_ablation' if it doesn't exist
# output_directory = 'zcp_aggr_ablation_bold'
output_directory = 'zcp_aggr_ablation_bold_clean'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

subsect_data = data.iloc[:, 12:]
# iterate through subsect_data, and make a list of accuracies, by indexing the first element in each row that is not a string
accuracies = [row[row.apply(lambda x: not isinstance(x, str)).idxmax()] for index, row in subsect_data.iterrows()]

data = data.iloc[:, [1, 4, 13]]
# add accuracies to data
data['perplexity'] = accuracies
# multiply by 100
data['perplexity'] = data['perplexity']

# proxy_name_dict = {
#     "epenas": "EPE-NAS",
#     "fisher": "Fisher",
#     "grad_norm": "GradNorm",
#     "grasp": "GRASP",
#     "jacov": "Jacov",
#     "l2_norm": "L2Norm",
#     "nwot": "NWOT",
#     "plainact": "PlainAct",
#     "snip": "SNIP",
# }
proxy_name_dict = {
    "epenas": "EPE-NAS",
    "fisher": r"Fisher ($\frac{d\mathcal{L}}{dA}$, $A$)",
    "grad_norm": r"GradNorm ($\frac{d\mathcal{L}}{dA}$)",
    "grasp": r"GRASP ($\frac{d^{2}\mathcal{L}}{dA^{2}}$, $\frac{d\mathcal{L}}{dA}$, $A$)",
    "jacov": r"Jacov ($\frac{d\mathcal{L}}{dA}$)",
    "l2_norm": r"L2Norm ($A$)",
    "nwot": "NWOT",
    "plainact": r"PlainAct ($\frac{d\mathcal{L}}{dA}$, $A$)",
    "snip": "SNIP",
}

data.columns = ['sparsity', 'proxy', 'task', 'perplexity']
ord_proxies = ["l2_norm", "grad_norm", "jacov", "plainact", "fisher", "grasp"]
data = data[data['proxy'].isin(ord_proxies)]

mathname = [proxy_name_dict[proxy] for proxy in ord_proxies]
data['proxy'] = data['proxy'].replace(proxy_name_dict)

# remove task wte
data = data[data['task'] == 'wikitext']
# sort sparsity
data = data.sort_values('sparsity')

# Find the proxy with the lowest perplexity for each sparsity level
best_proxies = data.loc[data.groupby('sparsity')['perplexity'].idxmin()]

# Plot the data with sparsity on x axis, perplexity on y axis, and different proxies as different lines
fig, ax = plt.subplots(figsize=(10, 6))
# for key, grp in data.groupby('proxy'):
for key in mathname:
    grp = data[data['proxy'] == key]
    linewidth = 2
    linedashed = "--"
    if key in best_proxies['proxy'].values:
        linewidth = 4
        linedashed = "-"
    ax = grp.plot(ax=ax, kind='line', x='sparsity', y='perplexity', label=key, marker='o', markersize=6, linewidth=linewidth, linestyle=linedashed)
    
# Set y axis upper lim to 10% more than max
# plt.ylim(data['perplexity'].min() * 0.95, data['perplexity'].max() * 2)
plt.ylim(data['perplexity'].min() * 0.95, 100)
# Y Axis label accuracy, X axis label "Sparsity"
plt.title('Perplexity On WikiText2', fontsize=24)
plt.ylabel('Perplexity', fontsize=24)
plt.xlabel('Sparsity (%)', fontsize=24)

# plt.yscale('log')
# Set legend fontsize as 18
plt.legend(fontsize=18, ncols=2, loc='upper left')
plt.grid(True)
plt.tight_layout()
# save the plot
plt.savefig(os.path.join(output_directory, 'aggr_wikitext_pplx.pdf'))
