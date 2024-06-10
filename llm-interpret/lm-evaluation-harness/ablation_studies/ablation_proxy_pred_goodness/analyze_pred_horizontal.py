import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set plot styles for IEEE scientific standard
plt.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.6,
    'grid.linestyle': '--',
    'axes.grid': True,
    'grid.color': 'grey'
})

# Load the data from the CSV file
data = pd.read_csv('predictor_results.csv')
# Remove rows with nan in it
data = data.dropna()

proxy_name_dict = {
    "epenas": "EPE-NAS",
    "fisher": "Fisher",
    "grad_norm": "GradNorm",
    "grasp": "GRASP",
    "jacov": "Jacov",
    "nwot": "NWOT",
    "l2_norm": "L2Norm",
    "plainact": "PlainAct",
    "snip": "SNIP",
}

plt.figure(figsize=(10, 6 ))
# Use a more muted color palette
# colors = plt.cm.cividis(np.linspace(0, 1, len(proxy_name_dict)+1))
colors = plt.cm.viridis(np.linspace(0, 1, len(proxy_name_dict)+1))
# colors = plt.cm.inferno(np.linspace(0, 1, len(proxy_name_dict)+1))

# colors = plt.cm.magma(np.linspace(0, 1, len(proxy_name_dict)+1))
# colors = plt.cm.coolwarm(np.linspace(0, 1, len(proxy_name_dict)+1))

# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors = plt.cm.Paired(np.linspace(0, 1, len(proxy_name_dict)+1))

# Replace proxy names in zcp_metric
data['zcp_metric'] = data['zcp_metric'].map(proxy_name_dict)
# Sort by tau
data = data.sort_values('tau', ascending=True)
# Create horizontal bar chart
plt.barh(data['zcp_metric'], data['tau'], color=colors, zorder=3, edgecolor='black')

plt.xlabel('Average Tau', fontsize=24)
# plt.ylabel('Proxy Predictor', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=20)
plt.xlim(left=np.min(data['tau']) * 0.97, right=np.max(data['tau']) * 1.03)
plt.title('ShadowLLM Predictive Performance Across Proxies', fontsize=24)
plt.tight_layout()
plt.savefig('proxy_pred_goodness.pdf')
plt.show()
