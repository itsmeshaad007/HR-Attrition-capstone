#import modules
import pandas as pd  # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
%matplotlib inline 
from sklearn.preprocessing import LabelEncoder

import numpy as np 
import warnings
# filter warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm,model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

import pandas as pd

def load_data(filepath):
    """
    Load data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data.
    """
    data = pd.read_csv(filepath)
    return data

def rename_columns(data):
    """
    Rename columns in a DataFrame.

    Args:
        data (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        pandas.DataFrame: The DataFrame with renamed columns.
    """
    renamed_columns = {
        'Id': 'Id',
        'Age': 'Age',
        'Attrition': 'Attrition',
        'BusinessTravel': 'Business_Travel',
        'Department': 'Department',
        'DistanceFromHome': 'Distance_From_Home',
        'Education': 'Education',
        'EducationField': 'Education_Field',
        'EmployeeNumber': 'Employee_Number',
        'EnvironmentSatisfaction': 'Environment_Satisfaction',
        'Gender': 'Gender',
        'JobInvolvement': 'Job_Involvement',
        'JobRole': 'Job_Role',
        'JobSatisfaction': 'Job_Satisfaction',
        'MaritalStatus': 'Marital_Status',
        'MonthlyIncome': 'Monthly_Income',
        'NumCompaniesWorked': 'Number_Of_Companies_Worked',
        'OverTime': 'Over_Time',
        'PercentSalaryHike': 'Percent_Salary_Hike',
        'PerformanceRating': 'Performance_Rating',
        'StockOptionLevel': 'Stock_Option_Level',
        'TotalWorkingYears': 'Total_Working_Years',
        'TrainingTimesLastYear': 'Training_Times_Last_Year',
        'YearsAtCompany': 'Years_At_Company',
        'YearsInCurrentRole': 'Years_In_Current_Role',
        'YearsSinceLastPromotion': 'Years_Since_Last_Promotion',
        'YearsWithCurrManager': 'Years_With_Current_Manager',
        'CommunicationSkill': 'Communication_Skill',
        'Behaviour': 'Behaviour'
    }

    data = data.rename(columns=renamed_columns)
    return data

def preprocess_data(data):
    # select categorical columns
    column = ['Business_Travel', 'Department', 'Education', 'Education_Field', 'Environment_Satisfaction', 'Gender',
              'Job_Involvement', 'Job_Role', 'Job_Satisfaction', 'Marital_Status', 'Number_Of_Companies_Worked',
              'Over_Time', 'Percent_Salary_Hike', 'Performance_Rating', 'Stock_Option_Level', 'Training_Times_Last_Year',
              'Years_In_Current_Role', 'Years_Since_Last_Promotion', 'Years_With_Current_Manager',
              'Communication_Skill', 'Behaviour']

    # one-hot encode categorical columns
    df_cat = pd.get_dummies(data[column], columns=column, prefix=column)

    # select numerical columns
    columns = ['Age','Distance_From_Home','Employee_Number','Monthly_Income','Total_Working_Years','Years_At_Company','Attrition']
    df_int64 = data[columns]

    # concatenate numerical and categorical dataframes
    df_num = pd.concat([df_int64, df_cat], axis=1)
    return df_num

def create_model(X_train, X_test, y_train, y_test):
    # import necessary libraries
    
    # load data and split into training and testing sets
    X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test

    # define parameters to be tuned
    parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
    }

    # create a random forest classifier object
    rf = RandomForestClassifier(random_state=42)

    # initialising GridSearchCV
    cv = GridSearchCV(estimator = rf, param_grid=parameters, cv=3)
    cv.fit(X_train, y_train)

    # extract the best estimator from the grid search results
    best_estimator = cv.best_estimator_

    # extract the best parameters from the grid search results
    best_params = cv.best_params_

    return cv, best_estimator, best_params
    
def display_results(cv, y_test, X_test, best_params, best_estimator):

    # make predictions on the test set using the best estimator
    y_pred = best_estimator.predict(X_test)

    # calculate and print precision, recall, and F1-score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Best parameters:", best_params)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", cm)


def main():
    data = load_data('train.csv')
    data = rename_columns(data)
    df_num = preprocess_data(data)
    test_size = 0.3
    random_state=0
    
    y= df_num.Attrition
    x= df_num.drop(["Attrition"],axis= 1)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    
    cv, best_estimator, best_params = create_model(X_train, X_test, y_train, y_test)
    
    display_results(cv, y_test, X_test, best_params, best_estimator)

main()