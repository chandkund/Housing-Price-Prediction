# Housing-Price-Prediction

This repository contains a project on predicting housing prices using various machine learning algorithms. The project includes data cleaning, preprocessing, exploratory data analysis, normalization, standardization, and model building using multiple regression techniques.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data_Loading](#data_loading)
- [Data_Cleaning_and_Preprocessing](#data_cleaning_and_preprocessing)
- [Exploratory_Data_Analysis](#exploratory_data_analysis)
- [Normalization_and_Standardization](#normalization_and_standardization)
- [Modeling](#modeling)
- [Results](#results)
- [License](#license)
  
## Introduction

The Boston Housing Dataset is a famous dataset derived from the Boston Census Service, originally curated by Harrison and Rubinfeld in 1978. The dataset contains information collected by the U.S. Census Service concerning housing in the area of Boston, Massachusetts.

The dataset includes 506 instances with 14 attributes or features:

- **CRIM**: per capita crime rate by town
- **ZN**: proportion of residential land zoned for lots over 25,000 sq. ft.
- **INDUS**: proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: nitrogen oxides concentration (parts per 10 million)
- **RM**: average number of rooms per dwelling
- **AGE**: proportion of owner-occupied units built prior to 1940
- **DIS**: weighted distances to five Boston employment centers
- **RAD**: index of accessibility to radial highways
- **TAX**: full-value property tax rate per $10,000
- **PTRATIO**: pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town
- **LSTAT**: percentage of lower status of the population
- **MEDV**: median value of owner-occupied homes in $1000s (Target Variable)

  
This dataset is particularly useful for regression tasks where the goal is to predict the median value of  owner-occupied homes (MEDV) using the other 13 attributes.

## Installation
To run this project, you need to have Python and the following libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
  
You can install these libraries using pip:

     pip install pandas numpy scikit-learn seaborn matplotlib
  
OR 
Clone the repository and install the required libraries:

     git clone https://github.com/chandkund/Housing-Price-Prediction.git

## Data_Loading
First, load the dataset using pandas:
```python
import pandas as pd
raw_data = pd.read_csv("D:\\Data_Science_Project\\Project_2\\HousingData.csv")
df = raw_data.copy()
df.info()  # Check the dataset information
```

## Data_Cleaning_and_Preprocessing
The following steps are performed for data cleaning and preprocessing:

-  Handle missing values in numerical and categorical columns using SimpleImputer.
-  Visualize the cleaned data using box plots and histograms.
-  Normalize and standardize the data for better model performance.
-  
 ### Import relevant Libraries
 ```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

 # Numerical and Categorical columns
 numberical_col = ['CRIM', 'ZN', 'INDUS', 'AGE','LSTAT']
 categorical_cols = ['CHAS']

 # Imputing missing values
 numberical_imputer = SimpleImputer(strategy='mean')
 df[numberical_col] = numberical_imputer.fit_transform(df[numberical_col])
 categorical_imputer = SimpleImputer(strategy='most_frequent')
 df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

 # Normalization
 scaler = MinMaxScaler()
 df[['CRIM', 'ZN','TAX','B']] = scaler.fit_transform(df[['CRIM', 'ZN','TAX','B']])

 # Standardization
 scaled = StandardScaler()
 df[['CRIM', 'ZN','TAX','B']] = scaled.fit_transform(df[['CRIM', 'ZN','TAX','B']])
 ```

## Exploratory_Data_Analysis
The data is visualized using various plots to understand the distributions and relationships between variables 
 ```python
import seaborn as sns
import matplotlib.pyplot as plt

# Box plots
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
ax = ax.flatten()
for index, col in enumerate(df.columns):
    sns.boxplot(y=col, data=df, ax=ax[index])
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()

# Histograms with KDE
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
ax = ax.flatten()
for index, col in enumerate(df.columns):
        sns.histplot(df[col], ax=ax[index], kde=True)
        ax[index].set_title(col)
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()
```

## Normalization_and_Standardization
  Normalization and standardization are applied to scale the features for better model performance.
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Min-Max Normalization
scaler = MinMaxScaler()
df[['CRIM', 'ZN','TAX','B']] = scaler.fit_transform(df[['CRIM', 'ZN','TAX','B']])
# Standardization
scaled = StandardScaler()
df[['CRIM', 'ZN','TAX','B']] = scaled.fit_transform(df[['CRIM', 'ZN','TAX','B']])
```
## Modeling
Different regression models are built and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor

### Modeing
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
# Splitting the data
inputs = df.drop(columns=['MEDV', 'RAD'])
targets = df['MEDV']
X_train, X_test, Y_train, Y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)
# Linear Regression
model_1 = LinearRegression()
model_1.fit(X_train, Y_train)
pred = model_1.predict(X_test)
mse = mean_squared_error(Y_test, pred)
print(f"Linear Regression MSE: {mse}")

# Decision Tree Regressor
model_2 = DecisionTreeRegressor()
model_2.fit(X_train, Y_train)
pred = model_2.predict(X_test)
mse = mean_squared_error(Y_test, pred)
print(f"Decision Tree MSE: {mse}")

# Random Forest Regressor
model_3 = RandomForestRegressor()
model_3.fit(X_train, Y_train)
pred = model_3.predict(X_test)
mse = mean_squared_error(Y_test, pred)
print(f"Random Forest MSE: {mse}")

# Extra Trees Regressor
model_4 = ExtraTreesRegressor()
model_4.fit(X_train, Y_train)
pred = model_4.predict(X_test)
mse = mean_squared_error(Y_test, pred)
print(f"Extra Trees MSE: {mse}")
 ```

## Results
The models are evaluated based on Mean Squared Error (MSE). Below are the MSE results for each model:

- Linear Regression MSE :26.84
- Decision Tree Regressor MSE :15.75
- Random Forest Regressor MSE :6.05
- Extra Trees Regressor MSE : 10.64

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



