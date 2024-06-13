import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set plot styles for IEEE scientific standard
plt.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.6,
    'grid.linestyle': '--'
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
# different color for each bar, virdis colors
colors = plt.cm.viridis(np.linspace(0, 1, len(proxy_name_dict)+1))
# replace proxy names in zcp_metric
data['zcp_metric'] = data['zcp_metric'].map(proxy_name_dict)
# put DejaVu first, then ShadowLLM, then Full Seq. ShadowLLM
plt.grid(True, zorder=0)
# sort by tau
data = data.sort_values('tau', ascending=True)
plt.bar(data['zcp_metric'], data['tau'], color=colors, zorder=3)
plt.xlabel('Proxy Predictor', fontsize=24)
plt.ylabel('Average Tau', fontsize=24)
# make x axis labels smaller
plt.xticks(fontsize=16)
plt.ylim(bottom=np.min(data['tau']) * 0.97, top=np.max(data['tau']) * 1.03)
plt.title('Performance of Proxy Predictors', fontsize=24)
plt.tight_layout()
plt.savefig('proxy_pred_goodness.pdf')