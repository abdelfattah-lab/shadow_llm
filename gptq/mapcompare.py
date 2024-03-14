import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import kendalltau
import scipy
# add argparser
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Visualize the attention maps')
parser.add_argument('--bw', type=int, default=2, help='s')

args = parser.parse_args()

headactivation_granularity = False
headaggregate_granilarity = True

if headactivation_granularity:
    # Load the attention data
    data_2_bits = torch.load(f'./shadow_llm_data/quant_attns_{args.bw}_opt-350m_.pt')
    print(f'Loaded {args.bw}bit data')
    data_16_bits = torch.load('./shadow_llm_data/quant_attns_16_opt-350m_.pt')
    print("Loaded 16bit data")
    nsamples = min(len(data_2_bits), len(data_16_bits))
    nlayers = len(data_2_bits[0])
    # Create base directory
    base_dir = f'base{args.bw}'
    os.makedirs(base_dir, exist_ok=True)
    # For each sample, create a new image
    # Within the image, have nlayers rows, head as 16 columns
    # For each nlayers,head, calculate the 
    for k in range(nsamples):
        rankscan = [50, 70, 90, 95, 99]
        cos_sim_map = {l: [] for l in range(nlayers)}
        kdt_map = {l: [] for l in range(nlayers)}
        rankscan_map = {r: {l: [] for l in range(nlayers)} for r in rankscan}
        for l in range(nlayers):
            for head in range(16):
                kmx2b = data_2_bits[k][l][0].squeeze()[head]
                kmxfp = data_16_bits[k][l][0].squeeze()[head]
                kmx2bmax = torch.max(kmx2b, dim=1)[0]
                kmxfpmax = torch.max(kmxfp, dim=1)[0]
                kdt, _ = kendalltau(kmx2bmax, kmxfpmax)
                kdt_map[l].append(kdt)
                cos_sim = torch.nn.functional.cosine_similarity(kmx2bmax, kmxfpmax, dim=0)
                cos_sim_map[l].append(cos_sim.item())
                kmx2_array = kmx2bmax
                kmxf_array = kmxfpmax
                for x in rankscan:
                    threshold_kmx2 = np.percentile(kmx2_array, x)
                    threshold_kmxf = np.percentile(kmxf_array, x)
                    indices_small_kmx2 = np.where(kmx2_array > threshold_kmx2)[0]
                    indices_small_kmxf = np.where(kmxf_array > threshold_kmxf)[0]
                    ranks_kmx2 = scipy.stats.rankdata(kmx2_array)
                    ranks_kmxf = scipy.stats.rankdata(kmxf_array)
                    small_ranks_kmx2 = ranks_kmx2[indices_small_kmx2]
                    small_ranks_kmxf = ranks_kmxf[indices_small_kmxf]
                    if len(small_ranks_kmx2) < len(small_ranks_kmxf):
                        small_ranks_kmxf = small_ranks_kmxf[:len(small_ranks_kmx2)]
                    elif len(small_ranks_kmxf) < len(small_ranks_kmx2):
                        small_ranks_kmx2 = small_ranks_kmx2[:len(small_ranks_kmxf)]
                    kdt, _ = scipy.stats.kendalltau(small_ranks_kmx2, small_ranks_kmxf)
                    rankscan_map[x][l].append(kdt)
        cos_sim_df = pd.DataFrame.from_dict(cos_sim_map, orient='index')
        kdt_df = pd.DataFrame.from_dict(kdt_map, orient='index')
        sample_dir = os.path.join(base_dir, str(k))
        os.makedirs(sample_dir, exist_ok=True)
        fig, ax = plt.subplots()
        ax.set_title('Cosine Similarity Map')
        heatmap = ax.imshow(cos_sim_df, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        fig.colorbar(heatmap, ax=ax)
        plt.savefig(os.path.join(sample_dir, 'cos_sim_map.png'))
        plt.close()
        fig, ax = plt.subplots()
        ax.set_title('Kendall Tau Map')
        heatmap = ax.imshow(kdt_df, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        fig.colorbar(heatmap, ax=ax)
        plt.savefig(os.path.join(sample_dir, 'kdt_map.png'))
        plt.close()
        for r in rankscan:
            rankscan_mapdf = pd.DataFrame.from_dict(rankscan_map[r], orient='index')
            fig, ax = plt.subplots()
            ax.set_title(f'Rankscan Map - {r}%')
            heatmap = ax.imshow(rankscan_mapdf, vmin=0, vmax=1, cmap='viridis', aspect='auto')
            fig.colorbar(heatmap, ax=ax)
            plt.savefig(os.path.join(sample_dir, f'rankscan_map_{r}.png'))
            plt.close()

if headaggregate_granilarity:
    # Load the attention data
    data_2_bits = torch.load(f'./shadow_llm_data/quant_attns_{args.bw}_opt-1.3b_.pt')
    print(f'Loaded {args.bw}bit data')
    data_16_bits = torch.load('./shadow_llm_data/quant_attns_16_opt-1.3b_.pt')
    print("Loaded 16bit data")
    nsamples = min(len(data_2_bits), len(data_16_bits))
    nlayers = len(data_2_bits[0])
    # Create base directory
    base_dir = f'hagg{args.bw}'
    os.makedirs(base_dir, exist_ok=True)
    for k in range(nsamples):
        rankscan = [50, 70, 80, 90, 92, 94]
        cos_sim_map = {l: [] for l in range(nlayers)}
        kdt_map = {l: [] for l in range(nlayers)}
        rankscan_map = {r: {l: None for l in range(nlayers)} for r in rankscan}
        for l in range(nlayers):
            h_2b = data_2_bits[k][l][0].squeeze().abs().mean(dim=1).max(dim=1)[0].tolist()
            h_fp = data_16_bits[k][l][0].squeeze().abs().mean(dim=1).max(dim=1)[0].tolist()
            kdt, _ = kendalltau(h_2b, h_fp)
            kdt_map[l].append(kdt)
            cos_sim = torch.nn.functional.cosine_similarity(torch.tensor(h_2b), torch.tensor(h_fp), dim=0)
            cos_sim_map[l].append(cos_sim.item())
        # Sort h_2b and h_fp and get its sorted index
        h_2b_sortindex = np.argsort(h_2b)[::-1]
        h_fp_sortindex = np.argsort(h_fp)[::-1]
        accmeasure = {}
        # for the top r% of heads, check intersection of the sorted indices as accuracy for r in rankscan
        for r in rankscan:
            h_2b_top = h_2b_sortindex[:int(r / 100 * len(h_2b))]
            h_fp_top = h_fp_sortindex[:int(r / 100 * len(h_fp))]
            accmeasure[r] = len(set(h_2b_top).intersection(h_fp_top)) / len(h_2b_top)
        cos_sim_df = pd.DataFrame.from_dict(cos_sim_map, orient='index')
        kdt_df = pd.DataFrame.from_dict(kdt_map, orient='index')
        sample_dir = os.path.join(base_dir, str(k))
        os.makedirs(sample_dir, exist_ok=True)
        fig, ax = plt.subplots()
        ax.set_title('Cosine Similarity Map')
        heatmap = ax.imshow(cos_sim_df, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        fig.colorbar(heatmap, ax=ax)
        plt.savefig(os.path.join(sample_dir, 'cos_sim_map.png'))
        plt.close()
        fig, ax = plt.subplots()
        ax.set_title('Kendall Tau Map')
        heatmap = ax.imshow(kdt_df, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        fig.colorbar(heatmap, ax=ax)
        plt.savefig(os.path.join(sample_dir, 'kdt_map.png'))
        plt.close()
        # Make a lineplot of accmeasure with key as x axis value as y axis
        fig, ax = plt.subplots()
        ax.set_title('Accuracy of Top Heads prediction')
        ax.plot(accmeasure.keys(), accmeasure.values())
        # x axis label
        ax.set_xlabel('Top x% Percentage of Heads (By activation magnitude)')
        # y axis label
        ax.set_ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(sample_dir, 'accmeasure.png'))
        plt.close()
        exit(0)