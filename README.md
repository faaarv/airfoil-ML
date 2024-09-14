# **Predict Airfoil Lift/Drag Coefficient using ML**
## Introduction

In this notebook, the application of machine learning techniques to predict aerodynamic coefficients for airfoils is explored, specifically the lift and drag coefficients. Machine Learning offers a more efficient approach to predicting airfoil performance compared to traditional methods like computational fluid dynamics (CFD) simulations and wind tunnel testing. 

## Dataset
 The dataset utilized in this project is a comprehensive collection of airfoil data, available through: https://github.com/nasa/airfoil-learning.


### Dataset Contents

The dataset includes the following key components:

#### Airfoil Geometry
- Represented by (x, y) coordinates that define the shape of each airfoil.

#### XFOIL Results
XFOIL, a popular software for airfoil analysis, provides critical aerodynamic coefficients:
- Drag coefficient (Cd)
- Pressure drag coefficient (Cdp)
- Moment coefficient (Cm)
- Lift coefficient (Cl)
- Pressure coefficient (Cp) for both upper and lower surfaces

#### Flow Conditions
Various parameters that affect airfoil performance:
- Reynolds number (Re)
- Angle of attack (AoA)
- Ncrit (transition criterion parameter)


For simplicity, we work with Cd, Cl as labels and Re,AoA,Ncrit,y values as Features. A dataframe is created from original dataset. The x values are the same for all the airfoils so they are removed from the features. then the dataset is normalized using two scalers: MinMax and Standard.

Preprocessing is a crucial step in machine learning, that involves transforming raw data into a format suitable for modeling. It enhances the quality of the data and helps improve the performance and accuracy of the machine learning models.

## Preprocessing

Normalization is a preprocessing technique used to scale the features of the dataset so they have a similar range of values. 

The MinMax Scaler is a normalization technique that transforms features by scaling them to a specific range, typically between 0 and 1. This method preserves the relationships between the original data values and maintains the distribution of the data.

The Standard Scaler, also known as Z-score normalization, transforms the data such that the resulting distribution has a mean of 0 and a standard deviation of 1. 

## Model

 Multilayer Perceptron is used for developing a ML model. A MLP is composed of multiple layers of nodes, including an input layer, one or more hidden layers, and an output layer.

For robust evaluation, K-Fold Cross-Validation is used. This technique assesses the performance and generalizability of a model by dividing the dataset into K subsets, or folds, and then training and evaluating the model K times. In each iteration, one fold is used as the test set, while the remaining K-1 folds are used for training. This process is repeated K times, with each fold serving as the test set exactly once. The performance metrics from each iteration are then averaged to obtain a more reliable evaluation of the model's performance.

K-Fold Cross-Validation helps reduce the impact of dataset variability on model assessment, providing a more reliable estimate of how well a model is likely to perform on unseen data. It is a widely used technique in machine learning to ensure a comprehensive evaluation of a model's capabilities.

K-Fold Cross-Validation also aids in determining the optimal values of hyperparameters. The performance of different architectures is compared using the average metrics obtained through this cross-validation method. This variant of cross-validation improves upon the holdout method by ensuring that the model score does not depend on how the training and test sets are chosen. 

## Training

Different model settings are used, including variations in the number of hidden layers, the number of units per layer, learning rate, number of epochs, and batch size. The criteria for evaluating the model include the mean squared error (MSE), the standard deviation of the MSE, training time, and the R-squared (R²) metric. These metrics help in assessing the accuracy, stability, and efficiency of the model under different configurations.



## Results

[View the notebook](https://github.com/faaarv/airfoil-ML/blob/main/results.ipynb)
