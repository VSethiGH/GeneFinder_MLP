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
    - gene_importance_dist.png : Graph of importance score distribution (normal, tumor, both) and key line

  - Flags:
    - --config : config .yaml path (configs/config.yaml)
    - --dir : output file path (log/), where model and new files are stored
    - --list_size : How long should the file gene list be (307)

