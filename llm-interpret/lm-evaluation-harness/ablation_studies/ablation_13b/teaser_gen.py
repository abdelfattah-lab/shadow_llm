import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# Set plot styles for IEEE scientific standard
plt.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.6,
    'grid.linestyle': '--'
})

# Read csv called teaser_info.csv
df = pd.read_csv('teaser_info.csv')

unique_sparsity = [10, 20, 30, 40, 50, 60, 70, 80, 90]
cmap = plt.get_cmap('viridis', lut=len(unique_sparsity))

# Normalize sparsity for color mapping
norm = Normalize(vmin=min(unique_sparsity), vmax=max(unique_sparsity))
colors = cmap(norm(df['sparsity']))

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# ShadowLLM
ax.plot(df['latency_shadowllm'], df['mean_acc_shadowllm'], linestyle='--', color='gray', zorder=1)
sc1 = ax.scatter(df['latency_shadowllm'], df['mean_acc_shadowllm'], c=colors, marker='*', s=400, label='ShadowLLM', zorder=3)

# DejaVu
ax.plot(df['latency_dejavu'], df['mean_acc_dejavu'], linestyle='--', color='gray', zorder=1)
sc2 = ax.scatter(df['latency_dejavu'], df['mean_acc_dejavu'], c=colors, marker='o', s=200, label='DejaVu-Style', zorder=3)

# Labels and Title
ax.set_xlabel('Latency (ms)', fontsize=24)
ax.set_ylabel('Accuracy (%)', fontsize=24)
ax.set_title('Accuracy vs. Latency on OPT-13B', fontsize=24)

ax.grid(True)
# Legend
ax.legend(fontsize=20, loc='lower right')

# Adjust layout to make space for colorbar
plt.tight_layout(rect=[0, 0, 0.89, 1])


# Create a new axis for the colorbar with a custom position
cax = fig.add_axes([0.92, 0.18, 0.02, 0.7])  # [left, bottom, width, height]

# Create a cropped colormap for colorbar
cmap_cropped = plt.get_cmap('viridis')
norm_cropped = Normalize(vmin=30, vmax=70)

sm = plt.cm.ScalarMappable(cmap=cmap_cropped, norm=norm_cropped)
sm.set_array([])

# Colorbar
cbar = plt.colorbar(sm, cax=cax, label='Sparsity (%)')
cbar.set_label('Sparsity (%)', fontsize=24, labelpad=-70, y=0.5, rotation=90)
cbar.set_ticks([30, 40, 50, 60, 70])  # Set the desired tick locations

# Adjust layout to make space for colorbar
# plt.tight_layout()


# Save the plot
plt.savefig('teaser_gen.pdf')

    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from matplotlib.colors import Normalize

    # # Set plot styles for IEEE scientific standard
    # plt.rcParams.update({
    #     'font.size': 18,
    #     'lines.linewidth': 2,
    #     'lines.markersize': 8,
    #     'grid.alpha': 0.6,
    #     'grid.linestyle': '--'
    # })

    # # Read csv called teaser_info.csv
    # df = pd.read_csv('teaser_info.csv')

    # unique_sparsity = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # cmap = plt.get_cmap('viridis', lut=len(unique_sparsity))

    # # Normalize sparsity for color mapping
    # norm = Normalize(vmin=min(unique_sparsity), vmax=max(unique_sparsity))
    # colors = cmap(norm(df['sparsity']))

    # # Plotting
    # fig, ax = plt.subplots(figsize=(8, 6))

    # # ShadowLLM
    # ax.plot(df['latency_shadowllm'], df['mean_acc_shadowllm'], linestyle='--', color='gray', zorder=1)
    # sc1 = ax.scatter(df['latency_shadowllm'], df['mean_acc_shadowllm'], c=colors, marker='*', s=400, label='ShadowLLM', zorder=3)


    # # DejaVu
    # ax.plot(df['latency_dejavu'], df['mean_acc_dejavu'], linestyle='--', color='gray', zorder=1)
    # sc2 = ax.scatter(df['latency_dejavu'], df['mean_acc_dejavu'], c=colors, marker='o', s=200, label='DejaVu', zorder=3)


    # # Adjust colorbar size

    # # Labels and Title
    # ax.set_xlabel('Latency (ms)', fontsize=24)
    # ax.set_ylabel('Accuracy (%)', fontsize=24)
    # ax.set_title('Latency vs. Accuracy on OPT-13B', fontsize=24)

    # ax.grid(True)
    # # Legend
    # ax.legend(fontsize=22, loc='lower right')

    # plt.tight_layout()
    # # Create a cropped colormap for colorbar
    # cmap_cropped = plt.get_cmap('viridis')
    # norm_cropped = Normalize(vmin=30, vmax=70)

    # sm = plt.cm.ScalarMappable(cmap=cmap_cropped, norm=norm_cropped)
    # sm.set_array([])

    # # Colorbar
    # cbar = plt.colorbar(sm, ax=ax, label='Sparsity (%)')
    # # cbar.set_label('Sparsity (%)', fontsize=24)
    # cbar.set_label('Sparsity (%)', fontsize=24, labelpad=-70, y=0.5, rotation=90)
    # cbar.set_ticks([30, 40, 50, 60, 70])  # Set the desired tick locations

    # plt.tight_layout()

    # # Save the plot
    # plt.savefig('teaser_gen.pdf')
