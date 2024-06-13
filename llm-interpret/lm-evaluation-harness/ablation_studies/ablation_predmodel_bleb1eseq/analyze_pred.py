
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

# 1. Which emb_style is best across data-sets?
# Calculating mean tau for each emb_style across datasets
best_emb_style = data.groupby('emb_style')['tau'].mean().reset_index()

plt.figure(figsize=(10, 6))
# different color for each bar, viridis colors
colors = plt.cm.viridis(np.linspace(0, 1, len(best_emb_style['emb_style'])+1))
emb_style_map = {
    "ble": "Full Seq.\nShadowLLM",
    "b1e": "ShadowLLM",
    "b1e_seq": "DejaVu\nStyle"
}
# replace emb_style with their full names
best_emb_style['emb_style'] = best_emb_style['emb_style'].replace(emb_style_map)
# sort by tau
best_emb_style = best_emb_style.sort_values('tau', ascending=True)

# Create horizontal bar chart
bars = plt.barh(best_emb_style['emb_style'], best_emb_style['tau'], color=colors[:len(best_emb_style)], zorder=3, edgecolor='black')

plt.xlabel('Average Spearman-Rho', fontsize=24)
# plt.ylabel('Model Style', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=20)
plt.xlim(left=np.min(best_emb_style['tau']) * 0.97, right=np.max(best_emb_style['tau']) * 1.03)
plt.title('Effectiveness Of Predictors Across Tasks', fontsize=24)
plt.tight_layout()
plt.savefig('best_model_design.pdf')
plt.show()
plt.clf()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Set plot styles for IEEE scientific standard
# plt.rcParams.update({
#     'font.size': 18,
#     'lines.linewidth': 2,
#     'lines.markersize': 8,
#     'grid.alpha': 0.6,
#     'grid.linestyle': '--'
# })

# # Load the data from the CSV file
# data = pd.read_csv('predictor_results.csv')
# # Remove rows with nan in it
# data = data.dropna()

# # 1. Which emb_style is best across data-sets?
# # Calculating mean tau for each emb_style across datasets
# best_emb_style = data.groupby('emb_style')['tau'].mean().reset_index()

# plt.figure(figsize=(10, 6 ))
# # different color for each bar, virdis colors
# colors = plt.cm.viridis(np.linspace(0, 1, len(best_emb_style['emb_style'])+1))
# emb_style_map = {
#     "ble": "Full Seq. ShadowLLM",
#     "b1e": "ShadowLLM",
#     "b1e_seq": "DejaVu"
# }
# # replace emb_style with their full names
# best_emb_style['emb_style'] = best_emb_style['emb_style'].replace(emb_style_map)
# # put DejaVu first, then ShadowLLM, then Full Seq. ShadowLLM
# # put grid behind bars
# plt.grid(True, zorder=0)
# best_emb_style = best_emb_style.sort_values('tau', ascending=True)
# plt.bar(best_emb_style['emb_style'], best_emb_style['tau'], color=colors, zorder=3)
# plt.xlabel('Model Style', fontsize=24)
# plt.ylabel('Average Tau', fontsize=24)
# plt.ylim(bottom=np.min(best_emb_style['tau']) * 0.97, top=np.max(best_emb_style['tau']) * 1.03)
# plt.title('Predictive Performance of Models', fontsize=24)
# plt.tight_layout()
# plt.savefig('best_model_design.pdf')
# plt.clf()
