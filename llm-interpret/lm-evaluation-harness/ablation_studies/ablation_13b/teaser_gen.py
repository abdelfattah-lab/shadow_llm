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
fig, ax = plt.subplots(figsize=(8, 8))

# ShadowLLM
ax.plot(df['latency_shadowllm'], df['mean_acc_shadowllm'], linestyle='--', color='gray', zorder=1)
sc1 = ax.scatter(df['latency_shadowllm'], df['mean_acc_shadowllm'], c=colors, marker='*', s=400, label='ShadowLLM', zorder=3)


# DejaVu
ax.plot(df['latency_dejavu'], df['mean_acc_dejavu'], linestyle='--', color='gray', zorder=1)
sc2 = ax.scatter(df['latency_dejavu'], df['mean_acc_dejavu'], c=colors, marker='o', s=200, label='DejaVu', zorder=3)


# Create a cropped colormap for colorbar
cmap_cropped = plt.get_cmap('viridis')
norm_cropped = Normalize(vmin=30, vmax=70)

sm = plt.cm.ScalarMappable(cmap=cmap_cropped, norm=norm_cropped)
sm.set_array([])

# Colorbar
cbar = plt.colorbar(sm)
cbar.set_label('Sparsity (%)')
cbar.set_ticks([30, 40, 50, 60, 70])  # Set the desired tick locations

# Adjust colorbar size

# Labels and Title
ax.set_xlabel('Latency', fontsize=24)
ax.set_ylabel('Accuracy (%)', fontsize=24)
ax.set_title('Latency vs. Accuracy on OPT-13B', fontsize=24)

ax.grid(True)
# Legend
ax.legend(fontsize=22, loc='lower right')

plt.tight_layout()
# Save the plot
plt.savefig('teaser_gen.pdf')

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

    # # Read csv called teaser_info.csv
    # df = pd.read_csv('teaser_info.csv')

    # unique_sparsity = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # cmap = plt.get_cmap('viridis', lut=len(unique_sparsity))  # Add 2 extra spots to the viridis colormap

    # # Create a custom colormap to offset lower colors
    # colors_array = cmap(np.linspace(0.2, 1, len(unique_sparsity)))  # Offset the lower colors by starting at 0.

    # # Normalize sparsity for color mapping
    # norm = plt.Normalize(min(unique_sparsity), max(unique_sparsity))
    # colors = cmap(norm(df['sparsity']))
    # # Plotting
    # fig, ax = plt.subplots(figsize=(8, 8))

    # # ShadowLLM
    # sc1 = ax.scatter(df['latency_shadowllm'], df['mean_acc_shadowllm'], c=colors, marker='*', s=400, label='ShadowLLM')
    # ax.plot(df['latency_shadowllm'], df['mean_acc_shadowllm'], linestyle='--', color='grey')  # Add dotted lines

    # # DejaVu
    # sc2 = ax.scatter(df['latency_dejavu'], df['mean_acc_dejavu'], c=colors, marker='o', s=200, label='DejaVu')
    # ax.plot(df['latency_dejavu'], df['mean_acc_dejavu'], linestyle='--', color='grey')  # Add dotted lines

    # # Colorbar
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm)
    # cbar.set_label('Sparsity (%)')

    # # Limit the colorbar range displayed to 30-70
    # cbar.set_ticks(np.linspace(30, 70, len(unique_sparsity)//2))

    # # Make the colorbar smaller
    # cbar.ax.set_aspect(20)
    # # Labels and Title
    # ax.set_xlabel('Latency', fontsize=24)
    # ax.set_ylabel('Accuracy (%)', fontsize=24)
    # ax.set_title('Latency vs. Accuracy on OPT-13B', fontsize=24)

    # ax.grid(True)
    # # Legend
    # ax.legend(fontsize=22, loc='lower right')

    # # Save the plot
    # plt.savefig('teaser_gen.pdf')

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
# # read csv called teaser_info.csv
# df = pd.read_csv('teaser_info.csv')
# # Create DataFrame
# df = pd.DataFrame(df)

# # Define colormap
# cmap = plt.get_cmap('viridis')

# # Normalize sparsity for color mapping
# norm = plt.Normalize(df['sparsity'].min(), df['sparsity'].max())
# colors = cmap(norm(df['sparsity']))

# # Plotting
# fig, ax = plt.subplots(figsize=(8,8))

# # ShadowLLM
# sc1 = ax.scatter(df['latency_shadowllm'], df['mean_acc_shadowllm'], c=colors, marker='*', s=200, label='ShadowLLM')

# # DejaVu
# sc2 = ax.scatter(df['latency_dejavu'], df['mean_acc_dejavu'], c=colors, marker='o', s=100, label='DejaVu')

# # Colorbar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm)
# cbar.set_label('Sparsity (%)')

# # Labels and Title
# ax.set_xlabel('Latency', fontsize=24)
# ax.set_ylabel('Accuracy (%)', fontsize=24)
# ax.set_title('Latency vs. Accuracy on OPT-13B', fontsize=20)

# ax.grid(True)
# # Legend
# ax.legend()

# plt.savefig('teaser_gen.pdf')