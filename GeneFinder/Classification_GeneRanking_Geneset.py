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

from sklearn.metrics import silhouette_score

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
import matplotlib.pyplot as plt

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

def load_hallmark_geneset(gmt_path):
    genesets = {}
    with open (gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            set_name = parts[0].strip()
            # 0 --> Name
            # 1 --> URL
            # 2 --> All the genes
            genes = [g.strip() for g in parts[2:] 
                if g.strip() != '']
            genesets[set_name] = genes
    
    return genesets

def create_chunk_sets(genesets, genes_index, chunk_size, X_train, y_train):
    chunks = []

    # Go through each hallmark set
    for set_name, genes in genesets.items():
        found_genes = [g for g in genes if g in genes_index]    # Looks for all genes in our data
        found_indices = [genes_index[g] for g in found_genes]   # Find the index for corresponding genes

        n_of_full_chunks = len(found_genes) // chunk_size

        for i in range(n_of_full_chunks):
            start = i * chunk_size
            end = start + chunk_size
            
            chunk_indices = found_indices[start:end]
            s_score = silhouette_score(X_train[:,chunk_indices],y_train)
        
            chunks.append({
                'set_name': set_name,
                'genes': found_genes[start:end],
                "indices": found_indices[start:end],
                "silhouette_score": s_score
            })

        remainder = len(found_genes) % chunk_size
        if remainder > 0:
            if n_of_full_chunks == 0:
                s_score = silhouette_score(X_train[:,found_indices],y_train)
                chunks.append({
                    'set_name': set_name,
                    'genes': found_genes,
                    'indices': found_indices,
                    'silhouette_score': s_score
                    })
            else:
                l_start = (n_of_full_chunks - 1) * chunk_size
                s_score = silhouette_score(X_train[:, chunk_indices], y_train)
                chunks[-1]['genes'] = found_genes[l_start:]
                chunks[-1]['indices'] = found_indices[l_start:]
                chunks[-1]['silhouette_score'] = s_score

    return chunks


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


    # Convert to tensor
    train_tensor = torch.FloatTensor(X_train)
    train_label_tensor = torch.LongTensor(y_train).reshape(-1, 1)
    test_tensor = torch.FloatTensor(X_test)
    test_label_tensor = torch.LongTensor(y_test).reshape(-1, 1)


    genesets = load_hallmark_geneset(args.hallmark)
    gene_names = list(train_data.df.columns)
    genes_index = {gene: i for i, gene in enumerate(gene_names)}


    chunks = create_chunk_sets(genesets, genes_index, 16, X_train, y_train)
    
    all_s_scores = np.array([chunk['silhouette_score'] for chunk in chunks])

    mean_s_scores = all_s_scores.mean()
    std_s_scores = all_s_scores.std()

    s_threshold = mean_s_scores + std_s_scores
    # s_threshold = np.percentile(all_s_scores, 75)


    plt.figure(figsize=(10, 6)) 
    plt.hist(all_s_scores, bins=100)
    plt.axvline(mean_s_scores, color='r', linestyle='--', label='Mean')
    plt.axvline(mean_s_scores + std_s_scores, color='g', linestyle='--', label='Mean + 1 STD')
    plt.axvline(mean_s_scores - std_s_scores, color='b', linestyle='--', label='Mean - 1 STD')


    plt.xlabel('Silhouette Score')
    plt.ylabel('Count')
    plt.title('Silhouette Score Distribution')
    plt.legend()
    save_path = os.path.join(args.dir, 'silhouette_score_distrib.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


    passed_chunks = [chunk for chunk in chunks if chunk['silhouette_score'] >= s_threshold]


    gene_scores = {}

    for chunk in passed_chunks:
        for gene in chunk['genes']:
            if gene not in gene_scores or chunk['silhouette_score'] > gene_scores[gene]:
                gene_scores[gene] = chunk['silhouette_score']



    ranked_genes = sorted(gene_scores.items(), key=lambda x : x[1], reverse = True)

    #top_genes = ranked_genes[:307]
    top_gene_names = [gene for gene, score in ranked_genes]

    top_gene_indices = [genes_index[gene]  for gene in top_gene_names]

    print("Post-Shilouette Length: ", len(top_gene_names))

    X_train = X_train[:, top_gene_indices]
    X_test = X_test[:, top_gene_indices]
    print(X_train.shape, X_test.shape)
    #y_train = y_train[:, top_gene_indices]
    #y_test = y_test[:, top_gene_indices]
    
    # Filter X_train and X_test (only top chunks genes)
    #X_train = X_train[top_gene_names]
    #X_test = X_test[top_gene_names]

    
    # Convert to tensor
    train_tensor = torch.FloatTensor(X_train)
    train_label_tensor = torch.LongTensor(y_train).reshape(-1, 1)
    test_tensor = torch.FloatTensor(X_test)
    test_label_tensor = torch.LongTensor(y_test).reshape(-1, 1)
    
    """

    save_path = os.path.join(args.dir, "Gene307_Hallmark.txt")
    with open(save_path, 'w') as f:
        f.write('Gene307_Hallmark\t' + '\t'.join(top_gene_names))

    print(f"Saved to {save_path}")
    """


    input_size = X_train.shape[1]        # Number of features (genes)
    hidden_size = [256, 64]              # <-- Need to look into
    num_classes = 2                      # Normal vs Tumor
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


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in testloader:
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
        val_loss /= len(testloader)
        model.train()


        print(f'Epoch [{epoch + 1}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
        if val_loss > avg_loss * 1.5:  # Val loss significantly higher than train loss
            print(f"⚠️  Possible overfitting detected at epoch {epoch + 1}")
        
        #print(f'Epoch [{epoch + 1}], Loss: {avg_loss:.4f}')

    print('Finished Training')

    
    # Create directory if it doesn't exist
    os.makedirs(args.dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(args.dir, 'hallmark_model.pth')
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
   

    print("Finding 307...")
    gene_list = []

    tumor_attrib = []
    normal_attrib = []

    # Goes through test data
    for i in range(len(test_tensor)):
        sample = test_tensor[i:i+1].clone().detach().float()
        label = test_label_tensor[i].item()
        sample.requires_grad_(True)                          
        model.zero_grad()
        output = model(sample)

        #tumor_class = 1
        #predicted_class = output.argmax(dim=1).item()
        #output[0, predicted_class].backward()

        (output[0, 1] - output[0, 0]).backward()

        rank = (sample.grad * sample).detach().cpu().numpy().squeeze()
        #gene_list.append(rank)
    
        if label == 1:
            tumor_attrib.append(rank)
        else:
            normal_attrib.append(rank)

    tumor_mean = np.mean(tumor_attrib, axis = 0)
    normal_mean = np.mean(normal_attrib, axis = 0)
    gene_importance = tumor_mean - normal_mean
    
    #gene_names = train_dataset.df.columns

    importance_df = pd.DataFrame({
        'gene': top_gene_names,
        'rating': gene_importance #gene_list_avg
    })

    print(len(top_gene_names), len(gene_importance))
    
    # Save Gene Ranking (Gene + Score)train_dataset.df.columns
    importance_df = importance_df.sort_values(by='rating', ascending=False)
    save_path = os.path.join(args.dir, 'gene_ranking_Hall.csv')
    importance_df.to_csv(save_path, sep='\t', index=False)

    mean_score = importance_df['rating'].mean()
    std_score = importance_df['rating'].std()
    
    threshold = mean_score + 1 * std_score

    
    top_307_df = importance_df[importance_df['rating'] >= threshold]['gene']
    #top_307_df = importance_df.nlargest(307, 'rating')['gene']

    save_path = os.path.join(args.dir, 'genelist_top_std_Hall.txt')
    top_307_df.to_csv(save_path, sep='\t', index=False, header=['#node1'])


    plt.figure(figsize=(10, 6))
    plt.hist(importance_df['rating'], bins=100)
    plt.axvline(mean_score, color='r', linestyle='--', label='Mean')
    plt.axvline(mean_score + std_score, color='g', linestyle='--', label='Mean + 1 STD')
    plt.axvline(mean_score - std_score, color='b', linestyle='--', label='Mean - 1 STD')


    plt.xlabel('Importance Score')
    plt.ylabel('Count')
    plt.title('Gene Importance Distribution')
    plt.legend()
    save_path = os.path.join(args.dir, 'gene_importance_distrib_geneset.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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
    parser.add_argument("--hallmark", type = str, default = "data/h.all.v2026.1.Hs.symbols.gmt")
    args = parser.parse_args()
    main(args)
