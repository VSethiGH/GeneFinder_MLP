# Data Processing

Please refer to the github repo: https://github.com/feltus/gembuild
Follow the instructions and Train / Test data (labels and log2 transform data) into the "data" directory

If a log file isn't made, the files below will create one...

# 19,738 Features 
In the "GeneFinder" directory, you will find the files...

- Classification_All_Genes.py
  - Utilizing all 19,738 Gene Features, creates an MLP model saved to "mlp_model.pth"

  - Flags:
    - --config : config .yaml path (configs/config.yaml)
    - --dir : output file path (log/), saves model to here


- GeneRanking_All_Genes.py
  - Calls model from Classification_All_Genes.py to rank the predicted gene set

  - Outputs:
    - gene_ranking.csv : CSV file ranking all features
    - genelist_top_size.txt : Top list_size (flag) gene names in the gene set
    - genelist.txt : Rank gene names in the gene set
    - gene_ranking_[threshold]_percentile.txt : Threshold Percentile of gene names in the gene set
      - Threshold = [25, 50, 75, 90, 95, 99]
    - gene_importance_dist.png : Graph of importance score distribution (normal, tumor, both) and lines for the threshold, mean, and std
    - log_gene_importance_dist.png : Distribution of logged gene importance 

  - Flags:
    - --config : config .yaml path (configs/config.yaml)
    - --dir : output file path (log/), where model and new files are stored
    - --list_size : How long should the file gene list be (307)
   
# Feature Selection Features  
- Classification_GeneRanking_Geneset.py
  - Utilizing a geneset from MSigDB, extract certain features, creates an MLP model saved to 'geneset_model.pth'
    
  - Outputs:
    - gene_ranking_Geneset.csv : CSV file ranking all features
    - genelist_top_std_Geneset.txt : Top list_size (1 std + mean) gene names in the gene set
    - gene_importance_dist.png : Graph of importance score distribution (normal, tumor, both) and lines for the threshold, mean, and std
    - log_gene_importance_dist.png : Distribution of logged gene importance
      
  - Flags:
    - --config : config .yaml path (configs/config.yaml)
    - --dir : output file path (log/), where model and new files are stored
    - --geneset : Location of MSigDB geneset ("data/h.all.v2026.1.Hs.symbols.gmt")

# Data Analysis

## Venn Diagrams & Fisher's Exact Test
- In the folder (Data Processing Files)...
- ComparePercentiles.py
  
  - Inputs:
    - Excel File: File where all the sheets live
    - Sheet 1 : Universal gene set (all genes)
    - Sheet 2 : All 19,738 Features in MLP extracted gene set
    - Sheet 3 : Core geneset 
    - Sheet 4 : Random geneset
    - Sheet 5 : Extra Geneset (Typically hallmark or geneset extracted from MSigDB)
    - Column : Column where all gene sets are at. Make sure they stay consistent throughout (default: A)

  - Creates Venn Diagrams to compare Sheet 2 (307 and each threshold) & 5 (Separately) against Sheet 3 & 4
  - Conducts Fisher's Exact Test to compare Sheet 2 (307 and each threshold), Sheet 5, Random (1 - 5) against Sheet 3 and 4
  
- Overlap.py
  - Inputs:
    - Excel File: File where all the sheets live
    - Sheet 1 : Where all genesets live (Each column is one gene set)
    - Column Names : gene set columns (comma-separated, e.g. C4,C6,Hallmark)
   
  - Creates "Overlap.png" comparing all columns in a Venn Diagram

## Gene Enrichment 
- When you have a selected gene set... go to https://toppgene.cchmc.org/
- Afterwards, click "ToppFun"
- Enter all genes in "Enrichment Gene Set" --> submit --> start
- "Download All" training results
- Copy and Paste the .txt content into an excel sheet
- Filter the excel sheet for when "q-value FDR B&H" is < e^-10
 

## Protein-Protein Interaction
- Protein.py
- Creates topology graph / emperical p-test accordingly 
  - Inputs:
    - Excel File: File where all the sheets live
    - Sheet 1 : Where all genesets live (Each column is one gene set)
    - Column Letter 1 : Where geneset 1 lives (Default G for C4)
    - Column Letter 2 : Where geneset 2 lives (Default I for C6)
  - Computes topology graph of Col 1, Col 2, Intersection (Col 1 and Col 2), Union - Intersection (Col 1 and Col 2)
  - Computes topology graph distribution for a random gene set 100 times for the corresponding size (emperical p-test)
