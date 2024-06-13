import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
csv_file = 'zcp_stats.csv'
df = pd.read_csv(csv_file)

# Prepare the data for plotting
plot_data = []
for _, row in df.iterrows():
    plot_data.append({
        'zcp_metric': row['zcp_metric'],
        'data_type': row['data_type'],
        'value': row['mean'],
        'metric': 'Mean'
    })
    plot_data.append({
        'zcp_metric': row['zcp_metric'],
        'data_type': row['data_type'],
        'value': row['min'],
        'metric': 'Min'
    })
    plot_data.append({
        'zcp_metric': row['zcp_metric'],
        'data_type': row['data_type'],
        'value': row['max'],
        'metric': 'Max'
    })
    plot_data.append({
        'zcp_metric': row['zcp_metric'],
        'data_type': row['data_type'],
        'value': row['25th_percentile'],
        'metric': '25th Percentile'
    })
    plot_data.append({
        'zcp_metric': row['zcp_metric'],
        'data_type': row['data_type'],
        'value': row['50th_percentile'],
        'metric': 'Median'
    })
    plot_data.append({
        'zcp_metric': row['zcp_metric'],
        'data_type': row['data_type'],
        'value': row['75th_percentile'],
        'metric': '75th Percentile'
    })

plot_df = pd.DataFrame(plot_data)

# Plot box plots
plt.figure(figsize=(14, 8))
sns.boxplot(x="zcp_metric", y="value", hue="data_type", data=plot_df)
plt.title("Comparison of Data Distributions Across Different ZCP Metrics")
plt.xlabel("ZCP Metric")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.legend(title="Data Type")
plt.tight_layout()
plt.savefig("zcp_statistics_ablation.pdf")
