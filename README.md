# HR-Attrition-capstone

## Problem Statement:
HR attrition is a critical issue for organizations, leading to significant financial losses, reduced productivity, and decreased morale among the remaining employees. Identifying the key factors that contribute to employee turnover and predicting the likelihood of attrition can help organizations proactively take steps to retain their employees and reduce the attrition rate. Therefore, the objective of this project is to develop a predictive model that can accurately forecast the likelihood of employee attrition based on various HR data points.

## Project Overview:
The project will involve analyzing HR data to determine the factors that contribute to employee attrition. The dataset will contain information about employee demographics, job characteristics, performance metrics, compensation and benefits, and other relevant variables. The dependent variable will be the attrition rate, which indicates whether an employee has left the organization or not. The model will be trained on historical data and will predict the likelihood of attrition for new employees.

## The project will involve the following steps:
Data Collection: The first step is to gather HR data from various sources and create a comprehensive dataset that includes all the relevant variables.

* Data Cleaning and Preparation: The data will be cleaned and preprocessed to remove any inconsistencies, missing values, and outliers. Feature engineering techniques will be applied to transform the data and create new features that may be useful in predicting attrition.

* Exploratory Data Analysis: The data will be visualized and analyzed to identify any patterns, trends, or correlations that may exist between the variables and the attrition rate.

* Model Development: Various machine learning algorithms will be applied to the dataset to develop a predictive model that can accurately forecast the likelihood of attrition. The model will be evaluated using appropriate performance metrics and fine-tuned using hyperparameter optimization techniques.

* Model Deployment: The final step is to deploy the model into a production environment, where it can be used to predict the likelihood of attrition for new employees and provide insights that can help organizations make informed decisions about employee retention.

This repository contains code for predicting employee attrition using machine learning techniques. The dataset used for training the model is train.csv, and the code is available in both Python script format (train_model.py) and Jupyter Notebook format (code.ipynb).

Dataset
The train.csv dataset contains information about employees in a company, including their age, job role, education level, and more. The target variable is "Attrition", which indicates whether or not an employee has left the company.

Running the Code
Dependencies
The following Python packages are required to run the code:

import pandas as pd  # for dataframes <br>
import matplotlib.pyplot as plt # for plotting graphs <br>
import seaborn as sns # for plotting graphs <br>
%matplotlib inline  <br>
from sklearn.preprocessing import LabelEncoder <br>
import numpy as np  <br>
import warnings <br>

warnings.filterwarnings('ignore') <br>
from sklearn.model_selection import train_test_split, cross_val_score <br>
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve <br>
from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network <br>
from sklearn.pipeline import make_pipeline <br>
from sklearn.linear_model import Ridge <br>
from sklearn.preprocessing import PolynomialFeatures <br>
from sklearn import svm,model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process <br>

import matplotlib.gridspec as gridspec <br>
from sklearn.preprocessing import LabelEncoder <br>
from sklearn.preprocessing import OneHotEncoder <br>
from sklearn.model_selection import train_test_split <br>
from sklearn.ensemble import RandomForestClassifier <br>
from sklearn.model_selection import GridSearchCV <br>
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score <br>

# Steps
Clone this repository to your local machine.
Install the required dependencies
Run the train_model.py script using the command python train_model.py. This will preprocess the data, train the model, and display the model's performance metrics.
Alternatively, open the FinalProject.ipynb notebook in Jupyter Notebook and run the cells in order.

# Conclusion
This code demonstrates a machine learning approach to predict employee attrition using a dataset of employee information. By analyzing various features of employees, we can predict which employees are most likely to leave the company, allowing the company to take proactive measures to retain those employees.

# Acknowledgement
Blogpost link: https://medium.com/@shadman.nw/hr-attrition-using-data-science-734c27485f92

https://www.gartner.com/en/human-resources/glossary/attrition#:~:text=Attrition%20is%20the%20departure%20of,%2C%20termination%2C%20death%20or%20retirement
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
https://pandas.pydata.org/docs/user_guide/10min.html
https://pandas.pydata.org/docs/
