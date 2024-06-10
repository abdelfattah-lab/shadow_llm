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
combined_data = combined_data[combined_data.iloc[:, 13]=="wikitext"]
# import pdb; pdb.set_trace()
# Create a new directory for saving the graphs
output_directory = 'ablation_13b'
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
predmap = {"predictorL": "ShadowLLM", "dejavu": "DejaVu"}
pmode_map = {"global": "Global", "perlayer": "Per-Layer"}

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
                # sort by sparsity
                subset = subset.sort_values('sparsity')
                if prune_mode == "perlayer":
                    # only sparsity > 40
                    subset = subset[subset['sparsity'] > 40]
                # only plot if f'{zcp}_{predmethod}' is in ["plainact_predictorL", "l2_norm_dejavu"]
                # if f'{zcp}_{predmethod}' in ["plainact_predictorL", "l2_norm_dejavu"]:
                ax.plot(subset['sparsity'], subset['perplexity'], marker='o', label=f'{predmap[predmethod]} ({zcpmap[zcp]})')
                    # ax.plot(subset['sparsity'], subset['perplexity'], marker=None, linestyle='-', label=f'{predmap[predmethod]} ({zcpmap[zcp]})')

    
    ax.set_title(f'{pmode_map[prune_mode]} Pruning On OPT-13B', fontsize=24)
    ax.set_xlabel('Sparsity (%)', fontsize=24)
    ax.set_ylabel('Perplexity', fontsize=24)
    ax.legend(fontsize=20, loc='upper left', ncol=1)
    ax.grid(True)
    # ax.set_yscale("log")
    # ax.tight_layout()

plt.tight_layout()
# Save the stacked plot as a PDF file
output_path = os.path.join(output_directory, 'stacked_perplexity_plots.pdf')
plt.savefig(output_path, bbox_inches='tight')

plt.cla()
plt.clf()
plt.close()



# Path to the directory containing CSV files
directory = 'all_ind_aggrzcp_res'

# Read and combine all CSV files
all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
data_list = [pd.read_csv(file, header=None) for file in all_files]

# Combine all data into a single DataFrame
combined_data = pd.concat(data_list, ignore_index=True)
combined_data = combined_data[combined_data.iloc[:, 13] == "wikitext"]

# Create a new directory for saving the graphs
output_directory = 'ablation_13b'
os.makedirs(output_directory, exist_ok=True)

# Set column names based on the provided data format
combined_data.columns = [
    'prune_mode', 'col1', 'col2', 'col3', 'zcp', 'col5', 'col6', 'col7', 'col8',
    'col9', 'sparsity', 'predmethod', 'col11', 'task', 'col13', 'col14', 'perplexity',
    'col16', 'col17'
]

# Extract unique values for zcp, prune_mode, and predmethod
zcps = combined_data['zcp'].unique()
predmethods = combined_data['predmethod'].unique()

zcpmap = {"plainact": "PlainAct", "l2_norm": "L2Norm"}
predmap = {"predictorL": "ShadowLLM", "dejavu": "DejaVu"}
pmode_map = {"global": "Global", "perlayer": "Per-Layer"}

# Set plot styles for IEEE scientific standard
plt.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.6,
    'grid.linestyle': '--'
})

plt.figure(figsize=(10, 5))

prune_mode = "global"
for zcp in zcps:
    for predmethod in predmethods:
        subset = combined_data[
            (combined_data['zcp'] == zcp) &
            (combined_data['prune_mode'] == prune_mode) &
            (combined_data['predmethod'] == predmethod)
        ]
        if not subset.empty:
            # less than sparsity 50
            subset = subset[subset['sparsity'] <= 50]
            # sort by sparsity
            subset = subset.sort_values('sparsity')
            plt.plot(subset['sparsity'], subset['perplexity'], marker='o', label=f'{predmap[predmethod]} ({zcpmap[zcp]})')

plt.title(f'{pmode_map[prune_mode]} Pruning On OPT-13B', fontsize=24)
plt.xlabel('Sparsity (%)', fontsize=24)
plt.ylabel('Perplexity', fontsize=24)
plt.legend(fontsize=20, loc='upper left', ncol=1)
plt.grid(True)

# Save the plot as a PDF file
output_path = os.path.join(output_directory, 'global_perplexity_plot.pdf')
plt.savefig(output_path, bbox_inches='tight')

plt.cla()
plt.clf()
plt.close()


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
combined_data = combined_data[combined_data.iloc[:, 13] != "wikitext"]
subsect_data = combined_data.iloc[:, 12:]
accuracies = [row[row.apply(lambda x: not isinstance(x, str)).idxmax()] for index, row in subsect_data.iterrows()]

# Create a new directory for saving the graphs
output_directory = 'ablation_13b'
os.makedirs(output_directory, exist_ok=True)

# Set column names based on the provided data format
combined_data.columns = [
    'prune_mode', 'col1', 'col2', 'col3', 'zcp', 'col5', 'col6', 'col7', 'col8',
    'col9', 'sparsity', 'predmethod', 'col11', 'task', 'col13', 'col14', 'accuracy',
    'col16', 'col17'
]

combined_data['accuracy'] = accuracies

# Extract unique values for zcp, prune_mode, and predmethod
zcps = combined_data['zcp'].unique()
tasklists = combined_data['task'].unique()

zcpmap = {"plainact": "PlainAct", "l2_norm": "L2Norm"}

# Set plot styles for IEEE scientific standard
plt.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.6,
    'grid.linestyle': '--'
})

fig, ax = plt.subplots(figsize=(10, 8))

accuracy_diff = []


task_map = {"piqa": "PIQA", "copa": "COPA", "openbookqa": "OpenbookQA", "winogrande": "Winogrande", "rte": "RTE", "hellaswag": "HellaSwag", "arc_easy": "ARC-Easy"}
for task_ in tasklists:
    subset_shadowllm = combined_data[
        (combined_data['zcp'] == 'plainact') &
        (combined_data['predmethod'] == 'predictorL') &
        (combined_data['task'] == task_)
    ]
    subset_dejavu = combined_data[
        (combined_data['zcp'] == 'l2_norm') &
        (combined_data['predmethod'] == 'dejavu') &
        (combined_data['task'] == task_)
    ]
    if not subset_shadowllm.empty and not subset_dejavu.empty:
        subset_shadowllm = subset_shadowllm.sort_values('sparsity')
        subset_dejavu = subset_dejavu.sort_values('sparsity')
        
        # Merge on sparsity to align values
        merged = pd.merge(subset_shadowllm[['sparsity', 'accuracy']], subset_dejavu[['sparsity', 'accuracy']], on='sparsity', how='outer', suffixes=('_shadowllm', '_dejavu')).fillna(0)
        
        acc_diff = (merged['accuracy_shadowllm'].values * 100) - (merged['accuracy_dejavu'].values * 100)
        accuracy_diff.append({
            'task': task_,
            'sparsity': merged['sparsity'].values,
            'accuracy_diff': acc_diff
        })

# Plotting the accuracy improvement
for data in accuracy_diff:
    ax.plot(data['sparsity'], data['accuracy_diff'], marker='o', label=f"{task_map[data['task']]}")


# Thin horizontal red line at accuracy_diff = 0 for all sparsities with no label
ax.axhline(y=0, color='r', linestyle='--')

ax.set_title('Accuracy Improvement of ShadowLLM over DejaVu on OPT-13B', fontsize=24)
ax.set_xlabel('Sparsity (%)', fontsize=24)
ax.set_ylabel('Accuracy Improvement (%)', fontsize=24)
ax.grid(True)

# Creating custom legends for the tasks
task_legend_labels = {task: task.upper() for task in tasklists}
ax.legend(fontsize=20, loc='upper left', ncol=1)

plt.tight_layout()
# Save the plot as a PDF file
output_path = os.path.join(output_directory, 'accuracy_improvement_plot.pdf')
plt.savefig(output_path, bbox_inches='tight')
plt.close()

from scipy.stats import gmean

# Calculate GEOMEAN and MEAN across tasks for each sparsity value
summary_data = []

# Define a function to calculate means and geometric means
def calculate_means(subset):
    return {
        'sparsity': sparsity,
        'mean_shadowllm': subset['accuracy_shadowllm'].mean() * 100,
        'gmean_shadowllm': gmean(subset['accuracy_shadowllm'].replace(0, 1)) * 100,  # replace 0 with 1 for gmean calculation
        'mean_dejavu': subset['accuracy_dejavu'].mean() * 100,
        'gmean_dejavu': gmean(subset['accuracy_dejavu'].replace(0, 1)) * 100  # replace 0 with 1 for gmean calculation
    }

# Iterate through unique sparsity values
sparsities = combined_data['sparsity'].unique()
for sparsity in sparsities:
    subset_shadowllm = combined_data[
        (combined_data['sparsity'] == sparsity) & 
        (combined_data['predmethod'] == 'predictorL')
    ]
    subset_dejavu = combined_data[
        (combined_data['sparsity'] == sparsity) & 
        (combined_data['predmethod'] == 'dejavu')
    ]

    if not subset_shadowllm.empty and not subset_dejavu.empty:
        merged = pd.merge(
            subset_shadowllm[['task', 'accuracy']].rename(columns={'accuracy': 'accuracy_shadowllm'}),
            subset_dejavu[['task', 'accuracy']].rename(columns={'accuracy': 'accuracy_dejavu'}),
            on='task',
            how='outer'
        ).fillna(0)  # Fill NaN with 0 for merging purposes

        means = calculate_means(merged)
        summary_data.append(means)

# Convert summary data to a DataFrame
summary_df = pd.DataFrame(summary_data)

# Save the DataFrame to a CSV file
summary_df.to_csv(os.path.join(output_directory, 'teaser_acc.csv'), index=False)

print("CSV file 'teaser_acc.csv' has been saved.")