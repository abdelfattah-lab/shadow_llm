import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import scipy
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.nn.utils.rnn import pad_sequence
import argparse
# python activation_predictor.py --dataset combined     --> done
# python activation_predictor.py --dataset copa
# python activation_predictor.py --dataset winogrande
# python activation_predictor.py --dataset openbookqa
# python activation_predictor.py --dataset piqa         --> done

parser = argparse.ArgumentParser()
parser.add_argument("--basemodel", type=str, default="opt-1.3b", help="opt-1.3b, opt-66b etc.")
parser.add_argument("--dataset", type=str, default="piqa", help="combined, copa, openbookqa, winogrande, piqa")
parser.add_argument("--execmode", type=str, default="train", help="train, headmodel")
args = parser.parse_args()


class SequenceModel(nn.Module):
    def __init__(self, embedding_dim=1024, output_dim=16):
        super(SequenceModel, self).__init__()
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, batch_first=True)
        
        # 2-layer MLP
        self.fc1 = nn.Linear(embedding_dim, embedding_dim//2)
        self.fc2 = nn.Linear(embedding_dim//2, output_dim)
    
    def generate_attention_mask(self, input_seq):
        # Assuming padding value is 0, create a mask for non-padded values
        # input_seq shape is assumed to be [B, L, E]
        mask = input_seq.ne(0).any(dim=2)  # Resulting shape [B, L]
        # Inverting the mask for attention: True for positions to attend, False for positions to ignore
        mask = ~mask  # Now mask is True where values should be ignored
        # No need to multiply by -1e9 here; we'll use this boolean mask directly
        return mask

    def forward(self, x):
        # Self-attention: [L, E] -> [L, L, E]
        attention_mask = self.generate_attention_mask(x)
        
        # Apply self-attention
        x, _ = self.self_attention(x, x, x, key_padding_mask=attention_mask)

        # Aggregating to [1, E]
        x = x.mean(dim=1, keepdim=True)
        
        # Passing through 2-layer MLP
        x = F.relu(self.fc1(x.squeeze(1)))
        x = self.fc2(x)
        
        return x

class SequenceDataset(Dataset):
    def __init__(self, data, layer_idx, max_seq_len=2048):
        self.data = data
        self.layer_idx = layer_idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_matrix = self.data[idx][0].squeeze()  # [L, E]
        target_activations = self.data[idx][1][self.layer_idx, :]  # [1, H]
        
        # Padding the sequence if necessary
        seq_len = input_matrix.shape[0]
        if seq_len < self.max_seq_len:
            padding = torch.zeros((self.max_seq_len - seq_len, input_matrix.shape[1]))
            input_matrix = torch.cat([input_matrix, padding], dim=0)
            
        return input_matrix, torch.tensor(target_activations, dtype=torch.float)

# Load data
base_path = f'zcps/opt-1.3b/'
with open(base_path + f'./fisher_trace_{args.dataset}_0.pkl', 'rb') as f:
    data = pickle.load(f)

import pdb; pdb.set_trace()
# Splitting data into 90% train and 10% test
# shuffle data dictionary by keys randomly
split_idx = int(len(data) * 0.9)
datavals = list(data.values())
# shuffle datavals list randomly using random package
random.shuffle(datavals)
train_data = datavals[:split_idx]
test_data = datavals[split_idx:]
max_seq_len = max([data[k][0].shape[1] for k in data])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
batch_size = 16

if args.execmode == "headmodel":
    num_dpoints = 2500
    # For each dataset, load the trace
    # load the predictor model respective to the dataset
    # Confusion matrix: X axis samples, Y axis layers, head-prediction correlation measured
    datasets = ["combined", "copa", "piqa", "openbookqa", "winogrande"][:3]
    overall_results = {}
    for dataset in datasets:
        with open(f'trace_inp_hact_{dataset}_original.pkl', 'rb') as f:
            data_analyze = pickle.load(f)
        perdata_headact = {x: [] for x in list(data_analyze.keys())[:num_dpoints]}
        for layer_idx in range(len(train_data[0][1])):
            # Model initialization
            embedding_dim = train_data[0][0].shape[-1]
            num_heads = train_data[0][1].shape[-1]  # Assuming target shape [D, H]
            model = SequenceModel(embedding_dim=embedding_dim, output_dim=num_heads).to(device)
            model.load_state_dict(torch.load(f'model_opt1.3b_{dataset}/model_layer_{layer_idx}.pt'))
            model.eval()
            for datakey in tqdm(list(data_analyze.keys())[:num_dpoints]):
                headactivation = data_analyze[datakey][1][layer_idx].squeeze()
                input_matrix = data_analyze[datakey][0].squeeze()
                input_matrix = input_matrix.unsqueeze(0)
                input_matrix = input_matrix.to(device)
                with torch.no_grad():
                    output = model(input_matrix.float())
                    output = output.detach().cpu().numpy().squeeze()
                    kdt = scipy.stats.kendalltau(output, headactivation)[0]
                    perdata_headact[datakey].append(kdt)
                    print(f"Layer {layer_idx}, Sample {datakey}, Kendall Tau: {kdt}")
        overall_results[dataset] = perdata_headact
        # Save the perdata_headact dictionary as a pickle file
        basedir = "samemodel_predictor"
        os.makedirs(basedir, exist_ok=True)
        with open(basedir + f'/perdata_headact_{dataset}.pkl', 'wb') as f:
            pickle.dump(perdata_headact, f)
        # Save the perdata_headact as a correlation matrix
        kdt_df = pd.DataFrame.from_dict(perdata_headact, orient='index')
        fig, ax = plt.subplots()
        ax.set_title('Kendall Tau Map')
        heatmap = ax.imshow(kdt_df, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        fig.colorbar(heatmap, ax=ax)
        plt.savefig(basedir + f'/kdt_map_{dataset}.png')
        plt.close()
    # Further, aggregate results per data-set
    # Confusion matrix: X axis data-set, Y axis layers, head-prediction correlation average per dataset
    perdataset_layerwise_kdt = {}
    for dataset in overall_results:
        perdataset_layerwise_kdt[dataset] = pd.DataFrame(overall_results[dataset]).T.mean().to_numpy().tolist()
    kdt_df = pd.DataFrame.from_dict(perdataset_layerwise_kdt, orient='index').T
    fig, ax = plt.subplots()
    ax.set_title('Kendall Tau Map')
    heatmap = ax.imshow(kdt_df, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    # Setting the labels for the axes.
    ax.set_xticks(range(len(kdt_df.columns)))
    ax.set_yticks(range(len(kdt_df.index)))
    ax.set_xticklabels(kdt_df.columns)
    ax.set_yticklabels(kdt_df.index)
    fig.colorbar(heatmap, ax=ax)
    plt.savefig(basedir + f'/kdt_map_perdataset.png')
    plt.close()

    print("modelling")
elif args.execmode == "combined_crosscheck":
    num_dpoints = 2500
    # For each dataset, load the trace
    # load the predictor model respective to the dataset
    # Confusion matrix: X axis samples, Y axis layers, head-prediction correlation measured
    datasets = ["combined", "copa", "piqa", "openbookqa", "winogrande"]
    overall_results = {}
    for dataset in datasets:
        with open(f'trace_inp_hact_{dataset}_original.pkl', 'rb') as f:
            data_analyze = pickle.load(f)
        perdata_headact = {x: [] for x in list(data_analyze.keys())[:num_dpoints]}
        for layer_idx in range(len(train_data[0][1])):
            # Model initialization
            embedding_dim = train_data[0][0].shape[-1]
            num_heads = train_data[0][1].shape[-1]  # Assuming target shape [D, H]
            model = SequenceModel(embedding_dim=embedding_dim, output_dim=num_heads).to(device)
            model.load_state_dict(torch.load(f'model_opt1.3b_combined/model_layer_{layer_idx}.pt'))
            model.eval()
            for datakey in tqdm(list(data_analyze.keys())[:num_dpoints]):
                headactivation = data_analyze[datakey][1][layer_idx].squeeze()
                input_matrix = data_analyze[datakey][0].squeeze()
                input_matrix = input_matrix.unsqueeze(0)
                input_matrix = input_matrix.to(device)
                with torch.no_grad():
                    output = model(input_matrix.float())
                    output = output.detach().cpu().numpy().squeeze()
                    kdt = scipy.stats.kendalltau(output, headactivation)[0]
                    perdata_headact[datakey].append(kdt)
                    print(f"Layer {layer_idx}, Sample {datakey}, Kendall Tau: {kdt}")
        overall_results[dataset] = perdata_headact
        # Save the perdata_headact dictionary as a pickle file
        basedir = f'predr_{args.basemodel}'
        os.makedirs(basedir, exist_ok=True)
        with open(basedir + f'/perdata_headact_{dataset}.pkl', 'wb') as f:
            pickle.dump(perdata_headact, f)
        # Save the perdata_headact as a correlation matrix
        kdt_df = pd.DataFrame.from_dict(perdata_headact, orient='index')
        fig, ax = plt.subplots()
        ax.set_title('Kendall Tau Map')
        heatmap = ax.imshow(kdt_df, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        fig.colorbar(heatmap, ax=ax)
        plt.savefig(basedir + f'/kdt_map_{dataset}.png')
        plt.close()
    # Further, aggregate results per data-set
    # Confusion matrix: X axis data-set, Y axis layers, head-prediction correlation average per dataset
    perdataset_layerwise_kdt = {}
    for dataset in overall_results:
        perdataset_layerwise_kdt[dataset] = pd.DataFrame(overall_results[dataset]).T.mean().to_numpy().tolist()
    kdt_df = pd.DataFrame.from_dict(perdataset_layerwise_kdt, orient='index').T
    fig, ax = plt.subplots()
    ax.set_title('Kendall Tau Map')
    heatmap = ax.imshow(kdt_df, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    # Setting the labels for the axes.
    ax.set_xticks(range(len(kdt_df.columns)))
    ax.set_yticks(range(len(kdt_df.index)))
    ax.set_xticklabels(kdt_df.columns)
    ax.set_yticklabels(kdt_df.index)
    fig.colorbar(heatmap, ax=ax)
    plt.savefig(basedir + f'/kdt_map_perdataset.png')
    plt.close()
    print("modelling")
elif args.execmode == "train":
    # Train and test models for each layer separately
    layer_kdt = {}
    for layer_idx in range(len(train_data[0][1])):
        # Prepare DataLoader for training and testing
        train_dataset = SequenceDataset(train_data, layer_idx, max_seq_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = SequenceDataset(test_data, layer_idx, max_seq_len)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Model initialization
        embedding_dim = train_data[0][0].shape[-1]
        num_heads = train_data[0][1].shape[-1]  # Assuming target shape [D, H]
        model = SequenceModel(embedding_dim=embedding_dim, output_dim=num_heads).to(device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        T_max = len(train_loader) * num_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in tqdm(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

            test_loss = 0
            kdt_register = []
            with torch.no_grad():
                track = 0
                for inputs, targets in test_loader:
                    if track == 10:
                        break
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    test_register_ = outputs.cpu().tolist()
                    target_register_ = targets.cpu().tolist()
                    kdt_register.extend([scipy.stats.kendalltau(test_register_[i], target_register_[i])[0] for i in range(len(test_register_))])
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    track += 1
            print(f"Epoch {epoch+1}, Test Loss: {test_loss}")

        # Testing loop
        model.eval()
        test_loss = 0
        kdt_register = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_register_ = outputs.cpu().tolist()
                target_register_ = targets.cpu().tolist()
                kdt_register.extend([scipy.stats.kendalltau(test_register_[i], target_register_[i])[0] for i in range(len(test_register_))])
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        layer_kdt[layer_idx] = kdt_register
        test_loss /= len(test_loader)
        print(f"Test Layer {layer_idx}, L2 Loss: {test_loss}")
        # Save the model with the name "model_layer_{layer_idx}.pt"
        basedir = f'model_{args.basemodel}_{args.dataset}'
        os.makedirs(basedir, exist_ok=True)
        torch.save(model.state_dict(), basedir + f'/model_layer_{layer_idx}.pt')

    # Visualization of results
    sample_dir = f'ft_{args.basemodel}_{args.dataset}'
    os.makedirs(sample_dir, exist_ok=True)

    for datap in range(len(layer_kdt[0])):
        kdtr = {l: layer_kdt[l][datap] for l in layer_kdt}
        kdt_df = pd.DataFrame.from_dict(kdtr, orient='index')
        fig, ax = plt.subplots()
        ax.set_title('Kendall Tau Map')
        heatmap = ax.imshow(kdt_df, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        fig.colorbar(heatmap, ax=ax)
        plt.savefig(os.path.join(sample_dir, f'kdt_map_{datap}.png'))
        plt.close()
