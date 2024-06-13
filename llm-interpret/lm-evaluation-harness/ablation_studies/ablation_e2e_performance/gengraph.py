import pandas as pd
import numpy as np

# Load the data from the CSV file
# data = pd.read_csv('a100_perfdata.csv')
data = pd.read_csv('a100_perffinal.csv')
data = data.dropna()

# Helper function to calculate percentage improvement
def calculate_percentage_improvement(df, baseline, comparison):
    baseline_mean = df[df['Mode'] == baseline]['Time (ms)'].mean()
    comparison_mean = df[df['Mode'] == comparison]['Time (ms)'].mean()
    improvement = ((baseline_mean - comparison_mean) / baseline_mean) * 100
    return improvement

# Analysis for Generation Length vs Time at 50% Sparsity
subset_50 = data[(data['Sparsity (%)'] == 50) & (data['prompt_length'] == 128) & (data['Model'] == "facebook/opt-30b")]
generation_modes = subset_50['Mode'].unique()

generation_results = {}
for baseline in generation_modes:
    generation_results[baseline] = {}
    for comparison in generation_modes:
        if baseline != comparison:
            generation_results[baseline][comparison] = calculate_percentage_improvement(subset_50, baseline, comparison)

# Analysis for Sparsity vs Time for prompt_length = 128 and Generation Length = 128
subset_128 = data[(data['prompt_length'] == 128) & (data['Generation Length (Tokens)'] == 128) & (data['Model'] == "facebook/opt-30b")]
sparsity_modes = subset_128['Mode'].unique()

sparsity_results = {}
for baseline in sparsity_modes:
    sparsity_results[baseline] = {}
    for comparison in sparsity_modes:
        if baseline != comparison:
            sparsity_results[baseline][comparison] = calculate_percentage_improvement(subset_128, baseline, comparison)

# Analysis for Different Models at 50% Sparsity, prompt_length = 128, Generation Length = 128
subset_cluster = data[(data['prompt_length'] == 128) & (data['Generation Length (Tokens)'] == 128) & (data['Sparsity (%)'] == 50)]
cluster_modes = subset_cluster['Mode'].unique()
cluster_models = subset_cluster['Model'].unique()

cluster_results = {}
for baseline in cluster_modes:
    cluster_results[baseline] = {}
    for comparison in cluster_modes:
        if baseline != comparison:
            improvements = []
            for model in cluster_models:
                model_data = subset_cluster[subset_cluster['Model'] == model]
                improvements.append(calculate_percentage_improvement(model_data, baseline, comparison))
            cluster_results[baseline][comparison] = max(improvements), min(improvements)

# Print the analysis results
def print_analysis(results, analysis_type):
    print(f"Analysis for {analysis_type}:")
    for baseline, comparisons in results.items():
        for comparison, improvement in comparisons.items():
            if isinstance(improvement, tuple):
                max_imp, min_imp = improvement
                print(f"{baseline} is UP TO {max_imp:.2f}% better than {comparison}, while being {min_imp:.2f}% worse.")
            else:
                print(f"{baseline} is {improvement:.2f}% better than {comparison}.")

print_analysis(generation_results, "Generation Length vs Time at 50% Sparsity")
print_analysis(sparsity_results, "Sparsity vs Time for prompt_length = 128 and Generation Length = 128")
print_analysis(cluster_results, "Different Models at 50% Sparsity, prompt_length = 128, Generation Length = 128")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
data = pd.read_csv('a100_perfdata.csv')
data = pd.read_csv('a100_perffinal.csv')
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
subset_50 = subset_50[subset_50['Model'] == "facebook/opt-30b"]
modes = subset_50['Mode'].unique()

mode_dict = {"dejavu": "DejaVu", "shadowllm": "ShadowLLM", "static": "Static"}


for mode in modes:
    mode_data = subset_50[subset_50['Mode'] == mode]
    # sort by sparsity
    mode_data = mode_data.sort_values(by='Sparsity (%)')
    # sort by generation length
    mode_data = mode_data.sort_values(by='Generation Length (Tokens)')
    ax.plot(mode_data['Generation Length (Tokens)'], mode_data['Time (ms)'] / 5, marker='o', label=mode_dict[mode])

ax.set_xlabel('Generation Length (Tokens)', fontsize=24)
ax.set_ylabel('Generation Time (ms)', fontsize=24)
ax.set_title('Generation Length vs Time For OPT-30B', fontsize=24)
ax.legend(fontsize=22)
ax.grid(True)
# log scale of 2
ax.set_xscale('log', base=2)
plt.tight_layout()
plt.savefig('generation_length_vs_time_50_sparsity.pdf')
plt.clf()

# Plot 2: Sparsity vs Time for prompt_length = 128 and Generation Length (Tokens) = 128
fig, ax = plt.subplots(figsize=(10, 6))
subset_128 = data[(data['prompt_length'] == 128) & (data['Generation Length (Tokens)'] == 128)]
subset_128 = subset_128[subset_128['Model'] == "facebook/opt-30b"]

# ax.plot(subset_128['Sparsity (%)'], subset_128['Time (ms)'], marker='o')

modes = subset_128['Mode'].unique()

mode_dict = {"dejavu": "DejaVu", "shadowllm": "ShadowLLM", "static": "Static"}

# divide each Time(sec) by corresponding  Generation Length (Tokens) to get Time per token
subset_128['Time (ms)'] = subset_128['Time (ms)'] / (5 * subset_128['Generation Length (Tokens)'])
for mode in modes:
    subset_128_m = subset_128[subset_128['Mode'] == mode]
    # sort by sparsity
    subset_128_m = subset_128_m.sort_values(by='Sparsity (%)')
    ax.plot(subset_128_m['Sparsity (%)'], subset_128_m['Time (ms)'], marker='o', label=mode_dict[mode])

ax.set_xlabel('Sparsity (%)', fontsize=24)
ax.set_ylabel('Per-Token Latency (ms)', fontsize=24)
ax.set_title('Sparsity vs Per-Token Latency for OPT-30B', fontsize=24)
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
    bar_times = [mode_data[mode_data['Model'] == model]['Time (ms)'].values[0]/5. for model in models]
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
