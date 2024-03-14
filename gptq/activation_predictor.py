import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import os

class SequenceModel(nn.Module):
    def __init__(self, seq_len=2048, embedding_dim=1024, output_dim=16, conv_out_channels=256):
        super(SequenceModel, self).__init__()
        
        # 1D Convolution layer to process each embedding across the sequence
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_out_channels, kernel_size=1)
        
        # Attention layer: using a simple linear layer to compute attention weights
        self.attention = nn.Linear(conv_out_channels, 1)
        
        # Fully connected layer to predict the final output
        self.fco = nn.Linear(conv_out_channels, conv_out_channels)
        # Fully connected layer to predict the final output
        self.fc = nn.Linear(conv_out_channels, output_dim)
    
    def forward(self, x):
        # Input x shape: (batch_size, seq_len, embedding_dim)
        # Transpose x to fit Conv1d input requirements: (batch_size, embedding_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Conv1d layer
        x = F.relu(self.conv1(x))
        
        # Transpose back to (batch_size, seq_len, conv_out_channels)
        x = x.transpose(1, 2)
        
        # Compute attention weights
        attention_weights = torch.softmax(self.attention(x), dim=1)
        
        # Apply attention weights (batch matrix multiplication)
        # (batch_size, 1, conv_out_channels)
        x = torch.bmm(attention_weights.transpose(1, 2), x).squeeze(1)
        
        # Fully connected layer to get the final output
        x = F.relu(self.fco(x))
        
        x = self.fc(x)
        
        return x

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Define a custom dataset
class SequenceDataset(Dataset):
    def __init__(self, dataset, layerid=0):
        self.layerid = layerid
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_matrix = np.asarray(self.dataset[idx]['inp'].tolist())  # (2048, 1024) input matrix
        target = np.asarray(self.dataset[idx][self.layerid][0])            # 16 element target list
        return torch.tensor(input_matrix, dtype=torch.float), torch.tensor(target, dtype=torch.float)

# Prepare DataLoader for training and testing
batch_size = 32  # Define your batch size

trainset = torch.load("./transformeremulator/quant_headacts_T_16_opt-350m_.pt")
testset = torch.load("./transformeremulator/quant_headacts_V_16_opt-350m_.pt")

print("Data loaded")

layer_kdt = {}
# import pdb; pdb.set_trace()
for layerid in [x for x in list(trainset[0].keys()) if isinstance(x, int)]:
    num_epochs = 2  # Define your number of epochs
    train_dataset = SequenceDataset(trainset, layerid)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = SequenceDataset(testset, layerid)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    print("Dataloader loaded")

    # Initialize the model, loss criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceModel().to(device)

    print("Model created")

    criterion = nn.MSELoss()  # L2 loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define your learning rate

    print("Optimizer created")

    # Total number of training steps per epoch is the length of the train loader
    T_max = len(train_loader) * num_epochs  # Defines the half-cycle for the scheduler

    # Initialize the scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)  # eta_min is the minimum LR


    print("Training loop started")
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
        
            # Step the LR scheduler after each batch
            scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

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
            kdt_register.append([scipy.stats.kendalltau(test_register_[i], target_register_[i])[0] for i in range(len(test_register_))])
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    # flatten kdt_register list of list
    kdt_register = [item for sublist in kdt_register for item in sublist]
    layer_kdt[layerid] = kdt_register
    test_loss /= len(test_loader)
    print(f"Test L2 Loss: {test_loss}")

# create director headmlpaggr
sample_dir = 'headmlpaggr'
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