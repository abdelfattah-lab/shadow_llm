import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
data = pd.read_csv('perfdata.csv')
data = data.dropna()

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

# Plot 1: Generation length vs Time for different modes at Sparsity (%) == 50
fig, ax = plt.subplots(figsize=(10, 6))
subset_50 = data[data['Sparsity (%)'] == 50]
subset_50 = subset_50[subset_50['prompt_length'] == 128]
subset_50 = subset_50[subset_50['Model'] == "facebook/opt-13b"]
modes = subset_50['Mode'].unique()

mode_dict = {"dejavu": "DejaVu", "shadowllm": "ShadowLLM", "static": "Static"}


for mode in modes:
    mode_data = subset_50[subset_50['Mode'] == mode]
    ax.plot(mode_data['Generation Length (Tokens)'], mode_data['Time (sec)'] / 5, marker='o', label=mode_dict[mode])

ax.set_xlabel('Generation Length (Tokens)', fontsize=24)
ax.set_ylabel('Generation Time (ms)', fontsize=24)
ax.set_title('Generation Length vs Time at 50% Sparsity', fontsize=24)
ax.legend(fontsize=22)
ax.grid(True)
plt.tight_layout()
plt.savefig('generation_length_vs_time_50_sparsity.pdf')
plt.clf()

# Plot 2: Sparsity vs Time for prompt_length = 128 and Generation Length (Tokens) = 128
fig, ax = plt.subplots(figsize=(10, 6))
subset_128 = data[(data['prompt_length'] == 128) & (data['Generation Length (Tokens)'] == 128)]
subset_128 = subset_128[subset_128['Model'] == "facebook/opt-13b"]

# ax.plot(subset_128['Sparsity (%)'], subset_128['Time (sec)'], marker='o')

modes = subset_128['Mode'].unique()

mode_dict = {"dejavu": "DejaVu", "shadowllm": "ShadowLLM", "static": "Static"}

# divide each Time(sec) by corresponding  Generation Length (Tokens) to get Time per token
subset_128['Time (sec)'] = subset_128['Time (sec)'] / (5 * subset_128['Generation Length (Tokens)'])
for mode in modes:
    mode_data = subset_128[subset_128['Mode'] == mode]
    ax.plot(mode_data['Sparsity (%)'], mode_data['Time (sec)'], marker='o', label=mode_dict[mode])

ax.set_xlabel('Sparsity (%)', fontsize=24)
ax.set_ylabel('Per-Token Latency (ms)', fontsize=24)
ax.set_title('Sparsity vs Per-Token Latency for OPT-13B', fontsize=24)
ax.legend(fontsize=22)
ax.grid(True)
plt.tight_layout()
plt.savefig('sparsity_vs_time_128_prompt_128_gen.pdf')
plt.clf()

# Plot 3: Clustered bar chart for different models at 50% Sparsity, prompt_length = 128, Generation Length (Tokens) = 128
subset_cluster = data[(data['prompt_length'] == 128) & (data['Generation Length (Tokens)'] == 128) & (data['Sparsity (%)'] == 50)]
models = subset_cluster['Model'].unique()
modes = subset_cluster['Mode'].unique()

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
bar_positions = np.arange(len(models))
colors = plt.cm.viridis(np.linspace(0, 1, len(modes) + 1))

for i, mode in enumerate(modes):
    mode_data = subset_cluster[subset_cluster['Mode'] == mode]
    bar_times = [mode_data[mode_data['Model'] == model]['Time (sec)'].values[0]/5. for model in models]
    ax.bar(bar_positions + i * bar_width, bar_times, bar_width, label=mode_dict[mode], color=colors[i], edgecolor='black', zorder=3)

ax.set_xlabel('Model', fontsize=24)
ax.set_ylabel('Time (ms)', fontsize=24)
ax.set_title('Generation Time For Different Models at 50% Sparsity', fontsize=24)
ax.set_xticks(bar_positions + bar_width * (len(modes) - 1) / 2)
ax.set_xticklabels([x.replace("facebook/", "").upper() for x in models], fontsize=20)
ax.legend(fontsize=22)
ax.grid(True, zorder=1)
plt.tight_layout()
plt.savefig('models_time_50_sparsity.pdf')
plt.clf()

plt.close('all')
