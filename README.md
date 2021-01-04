# Open Data Graph Ranking Models

This repository contains Jupyter notebooks that complement chapter 4 of my thesis titled "A system for disease-specific knowledge integration" towards a MSc in Data Science from National Centre for Scientific Research "Demokritos" and the University of the Peloponnese.

## Contents
 - **data_acquisition.ipynb**: Notebook that acquires required data from Neo4j using py2neo and dumps them to `fulldb-entities.json`.
 - **remove_label_extraction.ipynb**: Notebook that reads `combined.csv` and `test.csv` containing UMLS entities annotated for removal, changes labels marking entities as useful (value 1) or not useful (value 0) and dumps them to `removal_labels.json` and `test_sample.json`.  
 - **feature_selection.ipynb**: Notebook that implements our feature selection logic. Uses as input `fulldb-entities.json`, `removal_labels.json` and `test_sample.json`, creating `train_df.csv` and `test_df.csv` which contain the train and test set dataframes respectively.
 - **model_with_ranking_LogisticRegression.ipynb**: Notebook used to train, fine-tune and evaluate a Logistic Regression ranking ML algorithm using `train_df.csv` and `test_df.csv`.
 - **model_with_ranking_RankBoost.ipynb**: Notebook used to train, fine-tune and evaluate a RankBoost ranking ML algorithm using `train_df.csv` and `test_df.csv`.
 - **model_with_ranking_RankNet.ipynb**: Notebook used to train, fine-tune and evaluate a RankNet ranking ML algorithm using `train_df.csv` and `test_df.csv`.
 - **model_with_ranking_RankSVM.ipynb**: Notebook used to train, fine-tune and evaluate a RankSVM ranking ML algorithm using `train_df.csv` and `test_df.csv`.
 - **score_visual.ipynb**: Notebook used to visualize test scores of Logistic Regression, RankBoost, RankNet and RankSVM.
 
