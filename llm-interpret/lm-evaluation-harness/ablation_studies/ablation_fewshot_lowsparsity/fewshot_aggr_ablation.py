import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean

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

# Read and combine all CSV files
all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
data_list = [pd.read_csv(file, header=None) for file in all_files]

# Create a new directory for saving the graphs
output_directory = 'fewshot_ablation'
os.makedirs(output_directory, exist_ok=True)

# Combine all dataframes into a single dataframe
combined_data = pd.concat(data_list, ignore_index=True)

# save the dataframe to a csv file  in output directory
combined_data.to_csv(os.path.join(output_directory, 'combined_data.csv'), index=False)


# Extract the required columns from the combined dataframe
sparsity = combined_data.iloc[:, 1]
zcp = combined_data.iloc[:, 4]
fewshot = combined_data.iloc[:, 9].apply(lambda x: x.split("_")[-1].replace(".pkl", ""))
perplexity = combined_data.iloc[:, 16]

# Make a new dataframe from this
new_data = pd.DataFrame({'sparsity': sparsity, 'zcp': zcp, 'fewshot': fewshot, 'perplexity': perplexity})
# only keep zcp 'plainact'
# Get unique ZCP values
unique_zcp = new_data['zcp'].unique()

# Generate the Geomean plot
geomean_data = new_data.groupby(['sparsity', 'fewshot'])['perplexity'].apply(gmean).reset_index()
# Generate the mean
# geomean_data = new_data.groupby(['sparsity', 'fewshot'])['perplexity'].mean().reset_index()

plt.figure(figsize=(10, 6))
for fs in geomean_data['fewshot'].unique():
    fs_data = geomean_data[geomean_data['fewshot'] == fs]
    fs_data = fs_data.sort_values('sparsity')
    plt.plot(fs_data['sparsity'], fs_data['perplexity'], marker='o', label=f'{fs}-shot')

plt.xlabel('Sparsity (%)', fontsize=24)
plt.ylabel('Perplexity', fontsize=24)
plt.title('Perplexity On WikiText2 (OPT-1.3B)', fontsize=24)
plt.legend(fontsize=22, loc='upper center', ncol=3)
# plt.yscale('log')
plt.grid(True)
plt.tight_layout()

# Save the Geomean plot as a PDF file
output_path = os.path.join(output_directory, 'geomean_perplexity.pdf')
plt.savefig(output_path, bbox_inches='tight')
plt.close()