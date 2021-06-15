### --------------------------------------------------------------------------------------------------------
# Multiple Linear Regression - Boston House Prices
### --------------------------------------------------------------------------------------------------------
### Linear Regression:
- an **algorithm**:
    - based on **Supervised Learning**
    - where machines **learn with supervision**
    - **input data are labeled** and **expected output data is known**
    - used **to predict continuous (quantitative) numeric output value (y) based on given continuous (quantitative) numeric input variable(s) (x)**
    - where using the **linear relationship between numeric dependent output variable (y) (target) and independent input variable(s) (x) (predictor(s))**, we try **to find the best fit line (equation)** that can be used to make predictions
    - **best fit line** is known as **Regression line** and is represented by a **linear equation**:
                                        
                                        y = bn + an * xn, n>0
                              where,

                                        y - dependent variable we are trying to predict
                                        x - independent variables we are using to make predictions
                                        a, b - coefficients of Linear Regression equation (line)
                                        a - slope, which represents the effect x has on y
                                        b - intercept, which is a constant

- **regression analysis** is a **predictive modeling technique**
- **types** of Linear Regression:
  - **Simple Linear Regression** is a linear relationship between a single independent input variable (x) and corresponding dependent output variable (y)
  - **Multiple Linear Regression** is a linear relationship between 2 or more independent input variables (x) and corresponding output dependent variable (y)
- **examples** of Linear Regression problems:
  - House price
  - Weather forecast
  - Process optimization
  - Number of calls
  - Total sales
### --------------------------------------------------------------------------------------------------------
### Project Objective: Predicting the median value of owner-occupied homes in Boston
Create a model that allows to put in a few features of a house and returns back an estimate of the selling price of a house in various places in Boston. Information about the houses in Boston is in a built-in Boston dataset which is imported from Scikit-Learn datasets (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston).

The Boston Housing dataset is a derived from information collected by the U.S. Census Service concerning housing in the area of Boston MA.

The Boston dataset contains the following columns:
- **CRIM** per capita crime rate by town
- **ZN** proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS** proportion of non-retail business acres per town
- **CHAS** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- **NOX** nitric oxides concentration (parts per 10 million)
- **RM** average number of rooms per dwelling
- **AGE** proportion of owner-occupied units built prior to 1940
- **DIS** weighted distances to five Boston employment centres
- **RAD** index of accessibility to radial highways
- **TAX** full-value property-tax rate per 10,000 dollars
- **PTRATIO** pupil-teacher ratio by town
- **B** 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- **LSTAT** % lower status of the population
- **MEDV** Median value of owner-occupied homes in 1000’s dollars

### --------------------------------------------------------------------------------------------------------
### Table of Contents:
1. File Descriptions
2. Technologies Used
3. Structure of Notebook
4. Executive Summary

#### 1. File Descriptions
- Multiple Linear Regression - Boston_House_Prices.ipynb
- README.md

#### 2. Technologies Used
- Python
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-Learn

#### 3. Structure of Notebook
1. Import the Libraries
2. Load the Data
3. Set up the Data
4. Exploratory Data Analysis
    - 4.1 Check out the Data
    - 4.2 Data Visualization
5. Data Preprocessing and Feature Engineering
    - 5.1 Identify the variables
    - 5.2 Dealing with Missing values
    - 5.3 Dealing with Outliers
6. Train and Test the Linear Regression model
    - 6.1 Split the columns
    - 6.2 Split the data into Training dataset and Testing dataset
    - 6.3 Create the Linear Regression model
    - 6.4 Train / fit the Linear Regression model
    - 6.5 Calculate the coefficients of Linear Regression equation
    - 6.6 Predictions from the model on Testing data
    - 6.7 Evaluate the model on Testing data
7. Predict the label on new data

#### 4. Executive Summary
TBA
