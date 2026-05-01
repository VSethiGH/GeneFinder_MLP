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
import seaborn as sns


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

from scipy.stats import rankdata

from Classification  import MLPClassifier

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
    X_test = test_data.df.values
    y_test = test_data.label


    # First Index = Normal
    # Second Index = Tumor
    test_data.classes = ["Normal", "Tumor"]


    # Convert to tensor
    test_tensor = torch.FloatTensor(X_test)
    test_label_tensor = torch.LongTensor(y_test)

    input_size = X_test.shape[1]        # Number of features (genes)
    hidden_size = [256, 64]                     # <-- Need to look into
    num_classes = 2                      # Normal vs Tumori
    dropout = 0.6 

    # Check this right now

    # Create Model
    model = MLPClassifier(input_size = input_size, hidden_size = hidden_size, num_classes  = num_classes, dropout = dropout)
    
    mdl_path = os.path.join(args.dir, "mlp_model.pth")
    model.load_state_dict(torch.load(mdl_path))
    model.eval()

    gene_list = []

    tumor_attrib = []
    normal_attrib = []

    # Goes through test data
    for i in range(len(test_tensor)):
        sample = test_tensor[i:i+1].clone().detach().float()
        label = test_label_tensor[i].item()
        sample.requires_grad_(True)                            # Resets for each new gradient
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


    # Gets the average of each gene list
    
    # DF of Gene - Score
    gene_names = train_data.df.columns
    importance_df = pd.DataFrame({
        'gene': gene_names,
        'rating': gene_importance #gene_list_avg
    })

    # Save Gene Ranking (Gene + Score)
    importance_df = importance_df.sort_values(by='rating', ascending=False)
    save_path = os.path.join(args.dir, 'gene_ranking.csv')
    importance_df.to_csv(save_path, sep='\t', index=False)

    # Save top n  genes
    # top_genes_df = importance_df.head(args.list_size)['gene']

    mean_score = importance_df['rating'].mean()
    std_score = importance_df['rating'].std()
    
    top_307_df = importance_df.nlargest(args.list_size, 'rating')['gene']

    save_path = os.path.join(args.dir, 'genelist_top_size.txt')
    top_307_df.to_csv(save_path, sep='\t', index=False, header=['#node1'])

    """
    top_genes_df = importance_df[
    (importance_df['rating'] > mean_score + std_score)]['gene']

    save_path = os.path.join(args.dir, 'genelist.txt')
    top_genes_df.to_csv(save_path, sep='\t', index=False, header=['#node1'])

    with open(save_path, 'w') as f:
        f.write('genelist\t' +  '\t'.join(top_genes_df))

    """
    # Normal Plot

    percentiles = [25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(importance_df['rating'], percentiles)

    """
    for p, threshold in zip(percentiles, percentile_values):
        #threshold = np.percentile(importance_df['rating'], p)
        top_genes_df = importance_df[
            importance_df['rating'] >= threshold
        ]['gene']
    
        save_path = os.path.join(args.dir, 'gene_ranking_' + str(p) + '_percentile.txt')

        top_genes_df.to_csv(save_path, sep='\t', index=False, header=['#node1'])
    """

    plt.figure(figsize=(10, 6))
    plt.hist(importance_df['rating'], bins=100)
    plt.axvline(mean_score, color='r', linestyle='--', label='Mean')
    plt.axvline(mean_score + std_score, color='g', linestyle='--', label='Mean + 1 STD')
    plt.axvline(mean_score - std_score, color='b', linestyle='--', label='Mean - 1 STD')


    plt.hist(tumor_mean, bins=100, color='green', label='Tumor')
    plt.hist(normal_mean, bins=100, color='blue', label='Normal')

    #plt.axvline(np.mean(tumor_attrib), color='darkred', linestyle='-', label='Tumor Mean')
    #plt.axvline(np.mean(normal_attrib), color='darkblue', linestyle='-', label='Normal Mean')

    for p, val in zip(percentiles, percentile_values):
        plt.axvline(val, linestyle='--', label=f'{p}th percentile')

    plt.xlabel('Importance Score')
    plt.ylabel('Count')
    plt.title('Gene Importance Distribution')
    plt.legend()
    save_path = os.path.join(args.dir, 'gene_importance_dist.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Log Transform Plot

    #importance_df['log_rating'] = np.log2(importance_df['rating'])
    importance_df['log_rating'] = np.sign(importance_df['rating']) * np.log2(np.abs(importance_df['rating']) + 1)
    log_mean = importance_df['log_rating'].mean()
    log_std = importance_df['log_rating'].std()

    plt.figure(figsize=(10, 6))
    plt.hist(importance_df['rating'], bins=100)
    plt.axvline(log_mean, color='r', linestyle='--', label='Mean Log')
    plt.axvline(log_mean + log_std, color='g', linestyle='--', label='Mean + 1 STD')
    plt.axvline(log_mean - log_std, color='b', linestyle='--', label='Mean - 1 STD')
    plt.xlabel('Log - Importance Score')
    plt.ylabel('Count')
    plt.title('Logged - Gene Importance Distribution')
    plt.legend()
    save_path = os.path.join(args.dir, 'log_gene_importance_dist.png')
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
    parser.add_argument("--dir", type = str, default = "log/")
    parser.add_argument("--list_size", type = int, default = 307)
    args = parser.parse_args()
    main(args)
