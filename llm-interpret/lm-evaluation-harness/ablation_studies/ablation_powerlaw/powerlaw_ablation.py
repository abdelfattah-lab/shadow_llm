import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
import os
import seaborn as sns
import matplotlib.colors as mcolors
from tqdm import tqdm

# Set plot styles for IEEE scientific standard
plt.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.6,
    'grid.linestyle': '--'
})

filelist = [f for f in os.listdir('/home/ya255/projects/shadow_llm/llm-interpret/lm-evaluation-harness/zcps/opt-1.3b/') if "_trace_all_5.pkl" in f]

nlayers = 24
nheads = 32

def plot_rank_variance(data, num_elements, title, ylabel, filename):
    num_layers = nlayers
    elements_per_layer = num_elements // num_layers
    impidx = -2 if 'head' in title.lower() else -1
    
    rank_counts = np.zeros((num_elements, num_elements))

    for sample in data.values():
        activations = sample[impidx].flatten()
        ranks = np.argsort(activations).tolist()[::-1]  # Get ranks in descending order

        for i, rank in enumerate(ranks):
            rank_counts[rank, i] += 1

    rank_variances = np.var(rank_counts, axis=1)
    
    # Create color gradient for layers
    cmap = plt.get_cmap('viridis', num_layers+5)
    colors = cmap(np.linspace(0, 1, num_layers+5))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(num_layers):
        layer_start = i * elements_per_layer
        layer_end = (i + 1) * elements_per_layer
        ax.scatter(range(layer_start, layer_end), rank_variances[layer_start:layer_end], color=colors[i], marker='o', s=12)

    # Adding shaded regions for layer separation
    for i in range(num_layers):
        layer_start = i * elements_per_layer
        layer_end = (i + 1) * elements_per_layer
        ax.axvspan(layer_start, layer_end, color=colors[i], alpha=0.1)

    ax.set_xlabel('Head', fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_title(title, fontsize=24)
    # y axis log
    ax.set_yscale('log')
    # make plot fit inside
    plt.tight_layout()
    
    # Adding the colorbar
    norm = mcolors.Normalize(vmin=0, vmax=num_layers)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Layer')
    cbar.set_label('Layer', labelpad=-70, y=0.5, rotation=90, fontsize=22)
        # Calculate and add baseline lines
    random_variance = np.var(np.arange(num_elements))
    ax.axhline(y=random_variance, color='#B22222', linestyle='--')
    ax.text(num_elements-1, random_variance*0.9, 'Random Activation', color='#B22222', va='top', ha='right', fontweight='bold')

    from numpy.random import power

    alpha = 4.0  # Power-law exponent (can adjust slightly if needed)
    activations_power_law = power(alpha, num_elements)
    activations_power_law = activations_power_law / np.max(activations_power_law)  # Normalize
    power_law_rank_counts = np.zeros((num_elements, num_elements))
    for _ in range(1000):  # Simulate multiple samples to get an average behavior
        np.random.shuffle(activations_power_law)  # Shuffle to simulate different samples
        ranks = np.argsort(activations_power_law).tolist()[::-1]
        for i, rank in enumerate(ranks):
            power_law_rank_counts[rank, i] += 1

    power_law_variances = np.var(power_law_rank_counts, axis=1)
    power_law_variance = np.mean(power_law_variances)

    print(f'Power-Law Baseline Variance: {power_law_variance}')
    ax.axhline(y=power_law_variance, color='b', linestyle='--')
    ax.text(num_elements-1, power_law_variance, 'Power-Law Activation', color='b', va='bottom', ha='right', fontweight='bold')
    # tight fit
    plt.tight_layout()

    if 'head' in title.lower() == False:
        plt.savefig(filename, dpi=600)
    else:
        plt.savefig(filename, bbox_inches='tight')

for filename in tqdm(filelist):
    if "grad_norm" not in filename:
        continue
    with open('/home/ya255/projects/shadow_llm/llm-interpret/lm-evaluation-harness/zcps/opt-1.3b/' + filename, 'rb') as f:
        data = pickle.load(f)
    # make powerlaw_ablation directory
    os.makedirs("powerlaw_ablation", exist_ok=True)
    # Plot for heads
    plot_rank_variance(data, nlayers*nheads, 'Head Activation Rank Variance of OPT-1.3B', 'Rank Variance', f'powerlaw_ablation/{filename.replace("_trace_all_5.pkl","")}_1.3B_variance.pdf')
    print(f'powerlaw_ablation/{filename.replace("_trace_all_5.pkl","")}_1.3B_variance.pdf' + " saved")
