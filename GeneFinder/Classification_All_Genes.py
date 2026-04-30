import argparse
from omegaconf import OmegaConf
from diffusion import logger, dist_util
from diffusion.datasets import *
from diffusion.script_util import (
    model_and_diffusion_defaults,
    showdata,
    create_model_and_diffusion,
    get_silhouettescore,
)

from imblearn.over_sampling import SMOTE

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch.distributed as dist
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from diffusion.train_util import get_blob_logdir
import datetime
import os 

# Define the MLP model
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def main(args):
    print(args)

    basic_config = create_config()
    input_conf = OmegaConf.load(args.config)
    config = OmegaConf.merge(basic_config, input_conf)


    # Finds all the genes within the dataset 
    tempData  = pd.read_csv(config.data.train_path, sep='\t', index_col=0)
    gene_set = pd.DataFrame({"#node1": tempData.columns})

    # gene_set = gene_set.sample(n=10)
    
    gene_set.to_csv("all_genes.tsv", sep='\t', index=False)

    # Load Data 
    train_data, test_data = load_data(
                    train_path = config.data.train_path,
                    train_label_path = config.data.train_label_path,
                    test_path = config.data.test_path,
                    test_label_path = config.data.test_label_path,

                    gene_selection = 19738, # Max = 19738
                    class_cond=config.model.class_cond,
                    gene_set = "all_genes.tsv",
                    data_filter= config.data.filter,
    )
    

    # Convert all the data into dataframes so that the model can read it
    # X = Values
    # Y = Labels
    X_train = train_data.df.values
    y_train = train_data.label
   
    X_test = test_data.df.values
    y_test = test_data.label


    # First Index Should be GTEX, Second Index should be TCGA
    # This is to make sure that the model always has consistent labels 
    # If the first value is TCGA (Tumor), then swap the bits
    if train_data.classes[0] == "TCGA":
        y_train = 1 - y_train
        y_test = 1 - y_test

    # First Index = Normal
    # Second Index = Tumor
    train_data.classes = ["Normal", "Tumor"]
    test_data.classes = ["Normal", "Tumor"]

    # Apply SMOTE
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    # Convert to tensor
    train_tensor = torch.FloatTensor(X_train)
    train_label_tensor = torch.LongTensor(y_train).reshape(-1, 1)
    test_tensor = torch.FloatTensor(X_test)
    test_label_tensor = torch.LongTensor(y_test).reshape(-1, 1)

    input_size = X_train.shape[1]        # Number of features (genes)
    hidden_size = [256, 64]                     # <-- Need to look into
    num_classes = 2                      # Normal vs Tumori
    dropout = 0.6
    # Create Model
    model = MLPClassifier(input_size = input_size, hidden_size = hidden_size, num_classes  = num_classes, dropout = dropout)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Tensor / Dataloaders 
    batch_size = 32
    train_dataset = TensorDataset(train_tensor, train_label_tensor.squeeze())
    test_dataset = TensorDataset(test_tensor, test_label_tensor.squeeze())
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    for epoch in range(30): 
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)
        print(f'Epoch [{epoch + 1}], Loss: {avg_loss:.4f}')
    
    
    print('Finished Training')

    
    # Create directory if it doesn't exist
    os.makedirs(args.dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(args.dir, 'mlp_model.pth')
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to: {model_path}")

    # Train Model
    model.eval()
    size = len(testloader.dataset)
    num_batches = len(testloader)
    test_loss, correct = 0, 0

    
    all_preds = []
    all_labels = [] 

    with torch.no_grad():
        for X, y in testloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            all_preds.extend(pred.argmax(1).cpu().numpy())  
            all_labels.extend(y.cpu().numpy())
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    from sklearn.metrics import classification_report
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Tumor']))

def create_config():
    
    defaults = {
        "data":{
            "data_dir": "datasets",
            "cond": True,
        },
        "umap":{
            "n_neighbors": 90,
            "min_dist": 0.3,
        },
    }
    # defaults.update(model_and_diffusion_defaults())
    return defaults 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, default="configs/config.yaml")
    parser.add_argument("--dir", type = str, default = None)
    args = parser.parse_args()
    main(args)
