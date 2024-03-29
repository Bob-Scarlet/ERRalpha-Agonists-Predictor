Overview：

data/ contains the necessary dataset files;
models/ contains all baseline models and consensus model;
codes/ contains the codes of the construction of baseline models and consensos model.

Usage:

Step 1: Feature Preparation
Generate 5 combined features according to the method in the literature, with the specific format of the feature file as in the template.csv file, and name them 'rdkit-d+AP2D', 'rdkit-d+ECFP4', 'rdkit-d+EState', 'rdkit-d+FCFP4', 'rdkit-d+MACCS'. Save these files in the fingerprints and fingerprints_outer folders.

Step 2: Building the Baseline Model
The Baseline Model.py integrates cross-validation, testing, and external validation into one function and conducts 10 repeated experiments.
Running Baseline Model.py will generate the following folders: 
1. final_models, containing joblib files of the trained baseline models, 
2. results_all, containing results of cross-validation, testing, and external validation, 
3. results_10to1, containing the average and standard deviation of results from 10 repeated experiments.

Step 3: Building the Consensus Model
The Consensus Model also integrates cross-validation, testing, and external validation into one function.
Please note that the input for the consensus model is the output of 5 baseline models. Before running Consensus Model.py, ensure the correct paths to the joblib files of the 5 baseline models.
Running Consensus Model.py will generate the joblib file of the consensus model and its prediction results on the test set and external validation set.
