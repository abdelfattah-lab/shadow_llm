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
import seaborn as sns

# python emnlp_activation_predictor.py --dataset piqa --zcp_metric fisher --basemodel opt-1.3b --execmode train --emb_style b1e --rerun
# python emnlp_activation_predictor.py --dataset piqa --zcp_metric fisher --basemodel opt-1.3b --execmode train --emb_style b1e_seq --rerun
# python emnlp_activation_predictor.py --dataset piqa --zcp_metric fisher --basemodel opt-1.3b --execmode train --emb_style ble --rerun

parser = argparse.ArgumentParser()
parser.add_argument("--basemodel", type=str, default="opt-1.3b", help="opt-1.3b, opt-66b etc.")
parser.add_argument("--dataset", type=str, default="piqa", help="combined, copa, openbookqa, winogrande, piqa")
parser.add_argument("--dataset_cname", type=str, default="piqa", help="combined, copa, main, winogrande_xl, piqa")
parser.add_argument("--zcp_metric", type=str, default="fisher", help="fisher, l2_norm, plainact, etc.")
parser.add_argument("--execmode", type=str, default="train", help="train, headmodel")
parser.add_argument("--emb_style", type=str, default="b1e", help="b1e, b1eL, b1e_seq, ble")
parser.add_argument("--fewshot", type=int, default=0, help="0, 3, 5")
# boolean argument rerun
parser.add_argument("--rerun", action="store_true", help="rerun the model")
args = parser.parse_args()


dpath_style = args.emb_style if args.emb_style != "b1e_seq" else f"{args.emb_style}_{0}"
if os.path.exists(f'pred_models_{args.basemodel.split("-")[-1]}/{args.zcp_metric}/{args.dataset}/{dpath_style}.pt') and not args.rerun:
    # Write to file "already_done.log" which model exists
    with open("already_done.log", "a") as f:
        f.write(f"pred_models_{args.basemodel.split('-')[-1]}/{args.zcp_metric}/{args.dataset}/{dpath_style}.pt\n")
    exit(0)

class BLEPredModel(nn.Module):
    def __init__(self, embedding_dim=2048, output_dim=16):
        super(BLEPredModel, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, batch_first=True)
        self.fc1 = nn.Linear(embedding_dim, 2048)
        self.fc2 = nn.Linear(2048, output_dim)
    
    def generate_attention_mask(self, input_seq):
        mask = input_seq.ne(0).any(dim=2)  # Resulting shape [B, L]
        mask = ~mask  # Now mask is True where values should be ignored
        return mask

    def forward(self, x):
        attention_mask = self.generate_attention_mask(x)
        x, _ = self.self_attention(x, x, x, key_padding_mask=attention_mask)
        x = x.mean(dim=1, keepdim=True)
        x = F.relu(self.fc1(x.squeeze(1)))
        x = self.fc2(x)
        return x


class B1EPredModel(nn.Module):
    def __init__(self, embedding_dim=2048, output_dim=16):
        super(B1EPredModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 2048)
        self.fc2 = nn.Linear(2048, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x.squeeze()))
        x = self.fc2(x)
        return x


class B1ESeqPredModel(nn.Module):
    def __init__(self, embedding_dim=2048, output_dim=16):
        super(B1ESeqPredModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 2048)
        self.fc2 = nn.Linear(2048, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x.squeeze()))
        x = self.fc2(x)
        return x

class B1ESequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.valid_indices = self._filter_nan_indices()

    def _filter_nan_indices(self):
        valid_indices = []
        for idx in range(len(self.data)):
            input_matrix = self.data[idx][1][0].squeeze().detach()  # [E]
            # Normalize to 0-1
            input_matrix = (input_matrix - input_matrix.min()) / (input_matrix.max() - input_matrix.min())
            target_activations = self.data[idx][2].flatten().detach()  # [1, N*H]
            # Normalize to 0-1
            target_activations = (target_activations - target_activations.min()) / (target_activations.max() - target_activations.min())
            if not (torch.isnan(input_matrix).any() or torch.isnan(target_activations).any()):
                valid_indices.append(idx)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)
    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        input_matrix = self.data[idx][1][0].squeeze().detach()  # [E]
        # Normalize to 0-1
        input_matrix = (input_matrix - input_matrix.min()) / (input_matrix.max() - input_matrix.min())
        target_activations = self.data[idx][2].flatten().detach()  # [1, N*H]
        # Normalize to 0-1
        target_activations = (target_activations - target_activations.min()) / (target_activations.max() - target_activations.min())
        return input_matrix.float(), torch.tensor(target_activations, dtype=torch.float)

class B1ELSequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.valid_indices = self._filter_nan_indices()

    def _filter_nan_indices(self):
        valid_indices = []
        for idx in range(len(self.data)):
            input_matrix = self.data[idx][1][0].squeeze().detach()  # [E]
            # Normalize to 0-1
            input_matrix = (input_matrix - input_matrix.min()) / (input_matrix.max() - input_matrix.min())
            target_activations = self.data[idx][2].detach()  # [1, N*H]
            # Normalize to 0-1 for each row, keepdims=True
            target_activations = (target_activations - target_activations.min(keepdims=True, axis=1)[0]) / (target_activations.max(keepdims=True, axis=1)[0] - target_activations.min(keepdims=True, axis=1)[0])
            target_activations = target_activations.flatten()
            if not (torch.isnan(input_matrix).any() or torch.isnan(target_activations).any()):
                valid_indices.append(idx)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)
    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        input_matrix = self.data[idx][1][0].squeeze().detach()  # [E]
        # Normalize to 0-1
        input_matrix = (input_matrix - input_matrix.min()) / (input_matrix.max() - input_matrix.min())
        target_activations = self.data[idx][2].detach()  # [1, N*H]
        # Normalize to 0-1 for each row, keepdims=True
        target_activations = (target_activations - target_activations.min(keepdims=True, axis=1)[0]) / (target_activations.max(keepdims=True, axis=1)[0] - target_activations.min(keepdims=True, axis=1)[0])
        target_activations = target_activations.flatten()
        return input_matrix.float(), torch.tensor(target_activations, dtype=torch.float)

class B1ESeqSequenceDataset(Dataset):
    def __init__(self, data, seqlayer):
        self.data = data
        self.seqlayer = seqlayer
        self.valid_indices = self._filter_nan_indices()

    def _filter_nan_indices(self):
        valid_indices = []
        for idx in range(len(self.data)):
            input_matrix = self.data[idx][1][self.seqlayer].squeeze().detach()  # [E]
            # Normalize to 0-1
            input_matrix = (input_matrix - input_matrix.min()) / (input_matrix.max() - input_matrix.min())
            target_activations = self.data[idx][2][self.seqlayer].flatten().detach()  # [1, N*H]
            target_activations = (target_activations - target_activations.min()) / (target_activations.max() - target_activations.min())
            if not (torch.isnan(input_matrix).any() or torch.isnan(target_activations).any()):
                valid_indices.append(idx)
        return valid_indices
    def __len__(self):
        return len(self.valid_indices)
    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        input_matrix = self.data[idx][1][self.seqlayer].squeeze().detach()  # [E]
        # Normalize to 0-1
        input_matrix = (input_matrix - input_matrix.min()) / (input_matrix.max() - input_matrix.min())
        target_activations = self.data[idx][2][self.seqlayer].flatten().detach()  # [1, N*H]
        target_activations = (target_activations - target_activations.min()) / (target_activations.max() - target_activations.min())
        return input_matrix.float(), torch.tensor(target_activations, dtype=torch.float)
        
class FSeqSequenceDataset(Dataset):
    def __init__(self, data, max_seq_len=2048):
        self.data = data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_matrix = self.data[idx][0].squeeze().detach()  # [L, E]
        raise NotImplementedError("This is DISABLED. regenerate traces after removing [:, -1, :] from context_layer_val writes to first_embedding in base.py!")
        # Normalize to 0-1
        input_matrix = (input_matrix - input_matrix.min()) / (input_matrix.max() - input_matrix.min())
        target_activations = self.data[idx][2].flatten().detach()  # [1, N*H]
        seq_len = input_matrix.shape[0]
        if seq_len < self.max_seq_len:
            padding = torch.zeros((self.max_seq_len - seq_len, input_matrix.shape[1]))
            input_matrix = torch.cat([input_matrix, padding], dim=0)
        target_activations = (target_activations - target_activations.min()) / (target_activations.max() - target_activations.min())
        return input_matrix.float(), torch.tensor(target_activations, dtype=torch.float)

# Load data
base_path = f'zcps/{args.basemodel}/'
with open(base_path + f'./{args.zcp_metric}_trace_{args.dataset_cname}_{args.fewshot}.pkl', 'rb') as f:
    data = pickle.load(f)
# import pdb; pdb.set_trace()
debug_statistics = False
if debug_statistics:
    # make directory "zcp_statistics_ablation" if it doesnt exist
    ablation_path = "zcp_statistics_ablation"
    if not os.path.exists(ablation_path):
        os.makedirs(ablation_path)
    data_last = np.asarray([data[xgf][-1].flatten().cpu().numpy() for xgf in range(50)]).flatten()
    data_second_last = np.asarray([data[xgf][-2].flatten().cpu().numpy() for xgf in range(400)]).flatten()
    def print_statistics(data, name):
        print(f"Statistics for {name} on {args.zcp_metric}:")
        print(f"Mean: {np.mean(data)}")
        print(f"Std Dev: {np.std(data)}")
        print(f"Min: {np.min(data)}")
        print(f"Max: {np.max(data)}")
        print(f"25th percentile: {np.percentile(data, 25)}")
        print(f"50th percentile (median): {np.percentile(data, 50)}")
        print(f"75th percentile: {np.percentile(data, 75)}")
        print("\n")
    print_statistics(data_last, "FFN Activation")
    print_statistics(data_second_last, "Heads Activation")
    def plot_histogram(data, title):
        plt.hist(data, bins=2000, alpha=0.7)
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig(f'{ablation_path}/{args.zcp_metric}_{title}.pdf')
        plt.cla()
        plt.clf()
    plot_histogram(data_last, f'Distribution of FFN activations for {args.zcp_metric}')
    plot_histogram(data_second_last, f'Distribution of Heads activations for {args.zcp_metric}')
    def detect_outliers_iqr(data):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers
    outliers_last = detect_outliers_iqr(data_last)
    outliers_second_last = detect_outliers_iqr(data_second_last)
    print(f'Number of outliers in FFN for {args.zcp_metric}: {np.sum(outliers_last)}')
    print(f"Number of outliers in Heads for {args.zcp_metric}: {np.sum(outliers_second_last)}")
    def min_max_normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    data_last_normalized = min_max_normalize(data_last)
    data_second_last_normalized = min_max_normalize(data_second_last)
    plot_histogram(data_last_normalized, f'Normalized Distribution of FFN activations for {args.zcp_metric}')
    plot_histogram(data_second_last_normalized, f'Normalized Distribution of Heads activations for {args.zcp_metric}')
    # Prepare data for CSV
    stats = {
        "zcp_metric": args.zcp_metric,
        "data_type": ["FFN Activation", "Heads Activation"],
        "mean": [np.mean(data_last), np.mean(data_second_last)],
        "std_dev": [np.std(data_last), np.std(data_second_last)],
        "min": [np.min(data_last), np.min(data_second_last)],
        "max": [np.max(data_last), np.max(data_second_last)],
        "25th_percentile": [np.percentile(data_last, 25), np.percentile(data_second_last, 25)],
        "50th_percentile": [np.percentile(data_last, 50), np.percentile(data_second_last, 50)],
        "75th_percentile": [np.percentile(data_last, 75), np.percentile(data_second_last, 75)],
        "num_outliers": [np.sum(outliers_last), np.sum(outliers_second_last)]
    }
    df_stats = pd.DataFrame(stats)
    csv_file = f'{ablation_path}/zcp_stats.csv'
    if not os.path.exists(csv_file):
        df_stats.to_csv(csv_file, index=False)
    else:
        df_stats.to_csv(csv_file, mode='a', header=False, index=False)
    exit(0)

# Get i/o neuron information
nlayers = data[0][2].shape[0]
num_heads = data[0][2].shape[1]
total_heads = nlayers * num_heads
max_seq_len = max([data[k][0].shape[1] for k in data])
embedding_dim = data[0][1][0].shape[-1]

# Shuffle dataset
split_idx = int(len(data) * 0.8)
datavals = list(data.values())
random.shuffle(datavals)
train_data = datavals[:split_idx]
test_data = datavals[split_idx:]
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
batch_size = 32

if args.emb_style == 'b1e':
    # Create b1e dataset
    train_dataset = B1ESequenceDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    b1e_test_dataset = B1ESequenceDataset(test_data)
    test_loader = DataLoader(b1e_test_dataset, batch_size=batch_size, shuffle=False)
    model = B1EPredModel(embedding_dim=embedding_dim, output_dim=total_heads).to(device)
elif args.emb_style == 'b1e_seq':
    # Create b1eseq dataset
    b1e_seq_traindict = {}
    b1e_seq_testdict = {}
    for i in range(nlayers):
        b1e_seq_traindict[i] = B1ESeqSequenceDataset(train_data, i)
        b1e_seq_testdict[i] = B1ESeqSequenceDataset(test_data, i)
    b1e_seq_train_loader = {k: DataLoader(v, batch_size=batch_size, shuffle=True) for k, v in b1e_seq_traindict.items()}
    b1e_seq_test_loader = {k: DataLoader(v, batch_size=batch_size, shuffle=False) for k, v in b1e_seq_testdict.items()}
    model_dict = {k: B1ESeqPredModel(embedding_dim=embedding_dim, output_dim=num_heads).to(device) for k in range(nlayers)}
elif args.emb_style == "b1eL":
    # Create b1eL dataset
    train_dataset = B1ELSequenceDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    b1el_test_dataset = B1ELSequenceDataset(test_data)
    test_loader = DataLoader(b1el_test_dataset, batch_size=batch_size, shuffle=False)
    model = B1EPredModel(embedding_dim=embedding_dim, output_dim=total_heads).to(device)
elif args.emb_style == 'ble':
    # Create fseq dataset
    train_dataset = FSeqSequenceDataset(train_data, max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = FSeqSequenceDataset(test_data, max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = BLEPredModel(embedding_dim=embedding_dim, output_dim=total_heads).to(device)
else:
    raise ValueError("Invalid emb_style")

# make directory structure pred_models_{args.basemodel.split("-")[-1]}/{args.zcp_metric}/{args.dataset}/ if it doesnt exist
if not os.path.exists(f'pred_models_{args.basemodel.split("-")[-1]}/{args.zcp_metric}/{args.dataset}'):
    os.makedirs(f'pred_models_{args.basemodel.split("-")[-1]}/{args.zcp_metric}/{args.dataset}')
# Load model from state dict if it exists with name "pred_models_{args.basemodel.split("-")[-1]}/{args.zcp_metric}/{args.dataset}/{args.emb_style}.pt"
dpath_style = args.emb_style if args.emb_style != "b1e_seq" else f"{args.emb_style}_{0}"
if os.path.exists(f'pred_models_{args.basemodel.split("-")[-1]}/{args.zcp_metric}/{args.dataset}/{dpath_style}.pt') and not args.rerun:
    model.load_state_dict(torch.load(f'pred_models_{args.basemodel.split("-")[-1]}/{args.zcp_metric}/{args.dataset}/{dpath_style}.pt'))
else:
    if args.emb_style in ["b1e", "b1eL", "ble"]:
        # Train model from scratch
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        T_max = len(train_loader) * num_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
        # Train the model
        model.train()
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
                    kdt_register.extend([scipy.stats.spearmanr(test_register_[i], target_register_[i])[0] for i in range(len(test_register_))])
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    track += 1
            # Average kendall tau ranking across all inputs
            tau = sum(kdt_register)/len(kdt_register)
            print(f"Epoch {epoch+1}, Test Loss: {test_loss}, Spearman R: {tau}")

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
                kdt_register.extend([scipy.stats.spearmanr(test_register_[i], target_register_[i])[0] for i in range(len(test_register_))])
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        tau = sum(kdt_register)/len(kdt_register)
        print(f"Test Loss: {test_loss}, Spearman R: {tau}")
        # Save the test_loss and tau along with the args to a proper csv file in the pred_models_{args.basemodel.split("-")[-1]}/predictor_results.csv file
        if not os.path.exists(f'pred_models_{args.basemodel.split("-")[-1]}/predictor_results.csv'):
            with open(f'pred_models_{args.basemodel.split("-")[-1]}/predictor_results.csv', 'w') as f:
                f.write("dataset,fewshot,zcp_metric,emb_style,test_loss,tau\n")
        with open(f'pred_models_{args.basemodel.split("-")[-1]}/predictor_results.csv', 'a') as f:
            f.write(f"{args.dataset},{args.fewshot},{args.zcp_metric},{args.emb_style},{test_loss},{tau}\n")
        # Save predictor
        torch.save(model.state_dict(), f'pred_models_{args.basemodel.split("-")[-1]}/{args.zcp_metric}/{args.dataset}/{args.emb_style}.pt')
    elif args.emb_style == "b1e_seq":
        datapointwise_order = {idx: [[], []] for idx, (_, _) in enumerate(b1e_seq_test_loader[0])} # For each datapoint, construct a N*H order
        for curr_layer in range(nlayers):
            train_loader = b1e_seq_train_loader[curr_layer]
            test_loader = b1e_seq_test_loader[curr_layer]
            model = model_dict[curr_layer]
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001)
            T_max = len(train_loader) * num_epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
            model.train()
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
                with torch.no_grad():
                    track = 0
                    for inputs, targets in test_loader:
                        if track == 10:
                            break
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        test_register_ = outputs.cpu().tolist()
                        target_register_ = targets.cpu().tolist()
                        loss = criterion(outputs, targets)
                        test_loss += loss.item()
                        track += 1
                print(f"Epoch {epoch+1}, Test Loss: {test_loss}")
        
            model.eval()
            test_loss = 0
            true_orders = []
            pred_orders = []
            with torch.no_grad():
                for idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    test_register_ = outputs.cpu().tolist()
                    target_register_ = targets.cpu().tolist() # [16, N*H]
                    datapointwise_order[idx][0].append(test_register_)
                    datapointwise_order[idx][1].append(target_register_)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

            test_loss /= len(test_loader)
            print(f"Test L2 Loss: {test_loss}")
            # Save predictor
            torch.save(model.state_dict(), f'pred_models_{args.basemodel.split("-")[-1]}/{args.zcp_metric}/{args.dataset}/{args.emb_style}_{curr_layer}.pt')
        # Now, calculate the kendall tau for each datapoint (note that each datapoint actually has batch_size datapoints and average it out
        alldp_predorder = []
        alldp_trueorder = []
        for il in range(len(datapointwise_order)):
            np_nonlast_datapointwise_order = np.asarray(list(datapointwise_order.values())[il])
            np_nonlast_datapointwise_order = np_nonlast_datapointwise_order.swapaxes(1, 2)
            fshape, sshape = np_nonlast_datapointwise_order.shape[0], np_nonlast_datapointwise_order.shape[1]
            np_nonlast_datapointwise_order = np_nonlast_datapointwise_order.reshape(fshape, sshape, -1)
            predord = np_nonlast_datapointwise_order[0]
            trueord = np_nonlast_datapointwise_order[1]
            alldp_predorder.extend(predord)
            alldp_trueorder.extend(trueord)
        spearman_list = []
        for i in range(len(alldp_predorder)):
            spearman_list.append(scipy.stats.spearmanr(alldp_predorder[i], alldp_trueorder[i])[0])
        tau = sum(spearman_list)/len(spearman_list)
        print(f"Spearman R: {tau}")
        # Save the test_loss and tau along with the args to a proper csv file in the pred_models_{args.basemodel.split("-")[-1]}/predictor_results.csv file
        if not os.path.exists(f'pred_models_{args.basemodel.split("-")[-1]}/predictor_results.csv'):
            with open(f'pred_models_{args.basemodel.split("-")[-1]}/predictor_results.csv', 'w') as f:
                f.write("dataset,fewshot,zcp_metric,emb_style,test_loss,tau\n")
        with open(f'pred_models_{args.basemodel.split("-")[-1]}/predictor_results.csv', 'a') as f:
            f.write(f"{args.dataset},{args.fewshot},{args.zcp_metric},{args.emb_style},{0},{tau}\n")