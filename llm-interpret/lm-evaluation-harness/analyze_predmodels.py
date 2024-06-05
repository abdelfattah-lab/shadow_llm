import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


for msize in ["1.3b", "6.7b", "13b"]:
    try:
        # Load the data from the CSV file
        data = pd.read_csv(f'./pred_models_{msize}/predictor_results.csv')
        # Remove rows with nan in it
        data = data.dropna()

        # 1. Which emb_style is best across data-sets?
        # Calculating mean tau for each emb_style across datasets
        best_emb_style = data.groupby('emb_style')['tau'].mean().reset_index()

        plt.figure(figsize=(8, 5))
        plt.bar(best_emb_style['emb_style'], best_emb_style['tau'], color='skyblue')
        plt.xlabel('Embedding Style')
        plt.ylabel('Average Tau')
        plt.ylim(bottom=np.min(best_emb_style['tau']) * 0.95, top=np.max(best_emb_style['tau']) * 1.05)
        plt.title('Best Embedding Style Across Datasets')
        plt.savefig(f'./pred_models_{msize}/best_emb_style.png')
        plt.clf()

        # 2. Which metric is able to predict best across data-sets?
        # Calculating mean tau for each zcp_metric across datasets
        best_metric_pred = data.groupby('zcp_metric')['tau'].mean().reset_index()

        plt.figure(figsize=(8, 5))
        plt.bar(best_metric_pred['zcp_metric'], best_metric_pred['tau'], color='lightgreen')
        plt.xlabel('ZCP Metric')
        plt.ylabel('Average Tau')
        plt.ylim(bottom=np.min(best_metric_pred['tau']) * 0.95, top=np.max(best_metric_pred['tau']) * 1.05)
        plt.title('Best ZCP Metric for Prediction Across Datasets')
        plt.savefig(f'./pred_models_{msize}/best_metric_pred.png')
        plt.clf()

        # 3. Performance of metric across datasets, averaging embeddings where available
        # Using a scatter plot with different shapes for each metric
        data['emb_avg'] = data.groupby(['dataset', 'zcp_metric'])['tau'].transform('mean')
        data_unique = data.drop_duplicates(['dataset', 'zcp_metric', 'emb_avg'])

        cross_dset_pred = data_unique.groupby(['dataset', 'zcp_metric'])['emb_avg'].mean().unstack()

        fig, ax = plt.subplots(figsize=(10, 6))
        markers = ['o', 's', 'v', '^', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']
        colors = plt.cm.viridis(np.linspace(0, 1, len(cross_dset_pred.columns)))

        for i, metric in enumerate(cross_dset_pred.columns):
            ax.scatter(cross_dset_pred.index, cross_dset_pred[metric], label=metric, color=colors[i], marker=markers[i % len(markers)], s=100)

        ax.set_xlabel('Dataset')
        ax.set_ylabel('Average Tau')
        ax.set_title('Performance of Metrics Across Datasets')
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(f'./pred_models_{msize}/cross_dset_pred.png', bbox_inches='tight')
        plt.clf()
    except Exception as e:
        print(e)
        pass