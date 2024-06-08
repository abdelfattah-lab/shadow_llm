import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean

# Read the dataset from the CSV file
file_path = 'pred_models_1.3b/predictor_results.csv'
data = pd.read_csv(file_path)

# Filter the data where emb_style is 'b1e'
filtered_data = data[data['emb_style'] == 'b1e']

# Extract the required columns
filtered_data = filtered_data[['dataset', 'fewshot', 'zcp_metric', 'tau']]

# Get unique zcp_metric values
unique_zcp_metrics = filtered_data['zcp_metric'].unique()

# Create a directory for saving the graphs if it doesn't exist
output_directory = 'plots'
os.makedirs(output_directory, exist_ok=True)

# Generate plots for each unique zcp_metric
for zcp in unique_zcp_metrics:
    zcp_data = filtered_data[filtered_data['zcp_metric'] == zcp]

    plt.figure()
    for ds in zcp_data['dataset'].unique():
        ds_data = zcp_data[zcp_data['dataset'] == ds]
        # Sort ds_data by fewshot
        ds_data = ds_data.sort_values('fewshot')
        plt.plot(ds_data['fewshot'], ds_data['tau'], label=f'Dataset {ds}')

    plt.xlabel('Fewshot')
    plt.ylabel('Tau')
    plt.title(f'ZCP Metric: {zcp}')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PDF file
    output_path = os.path.join(output_directory, f'pred_{zcp}.pdf')
    plt.savefig(output_path)
    plt.close()



# Generate the Geomean plot across all zcp_metrics
geomean_data = filtered_data.groupby(['dataset', 'fewshot'])['tau'].apply(gmean).reset_index()

plt.figure()
for ds in geomean_data['dataset'].unique():
    ds_data = geomean_data[geomean_data['dataset'] == ds]
    # Sort ds_data by fewshot
    ds_data = ds_data.sort_values('fewshot')
    plt.plot(ds_data['fewshot'], ds_data['tau'], label=f'Dataset {ds}')

plt.xlabel('Fewshot')
plt.ylabel('Geometric Mean of Tau')
plt.title('Geometric Mean of Tau across ZCP Metrics')
plt.legend()
plt.grid(True)

# Save the Geomean plot as a PDF file
output_path = os.path.join(output_directory, 'pred_geomean.pdf')
plt.savefig(output_path)
plt.close()