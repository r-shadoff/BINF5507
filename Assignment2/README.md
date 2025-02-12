Regression and Classification Models
Written in VSCode

This repository contains an analysis of how regression and classification models can be tuned using hyperparameters using the UCI heart disease dataset.

scripts/main.ipynb contains code for:
- a linear regression model that predicts cholesterol levels and a plot showing the predicted versus true cholesterol values
- a logistic regression model that assigns a sample as having heart disease (1) or being healthy (0) and its accompanying ROC-AUC and precision recall curves
- a kNN model that assigns a sample as having heart disease (1) or being healthy (0) and its accompanying ROC-AUC and precision recall curves

scripts/data_preprocessor.py defines multiple data cleaning functions that are called upon in main.ipynb. A full description of data_preprocessor can be found in my Assignment1 repository's README.md file.

To run this project, download this repository OR ensure that data_preprocessor.ipynb and main.ipynb are within the folder you are running from, and the heart disease data is in its own folder labelled Data.
