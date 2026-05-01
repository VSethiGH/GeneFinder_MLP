**Data Processing**
Please refer to the github repo: https://github.com/feltus/gembuild
Follow the instructions and Train / Test data (labels and log2 transform data) into the "data" directory

If a log file isn't made, the files below will create one...

**19,738 Features**
In the "GeneFinder" directory, you will find the files...

- Classification_All_Genes.py
Utilizing all 19,738 Gene Features, creates an MLP model saved to "mlp_model.pth"

Flags:
--config : config .yaml path (configs/config.yaml)
--dir : output file path (log/), saves model to here


- GeneRanking_All_Genes.py
Calls model in Classification_All_Genes.py

Outputs:
gene_ranking.csv : CSV file ranking all features
genelist_top307.txt : Top 307 in gene 

Flags:
--config : config .yaml path (configs/config.yaml)
--dir : output file path (log/), where model and new files are stored
--list_size : How long should the file gene list be (307)

