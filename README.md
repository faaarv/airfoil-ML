# airfoil-ML
## Introduction

In this repository,  Airfoil's Lift and Drag Coefficient is predicted using Machine Learning

## Dataset
 The dataset used in this work is available at: https://github.com/nasa/airfoil-learning.

This dataset ( not normalized ) contains the geometry (x, y) of airfoils and the XFOIL results (Cd, Cdp, Cm , Cl , Cp for upper and lower surface) for various Reynolds numbers (Re), angles of attack (AoA), and Ncrit for each airfoil. 

After  removing unnecessary fields such as  Cp, Cdp, Cm, x. A dataframe is created from original dataset. The x values are the same for all the  airfoils so they are removed from the features. then the dataset is normalized using two scalers: MinMax and Standard.

Preprocessing is a crucial step in machine learning, that involves transforming raw data into a format suitable for modeling. It enhances the quality of the data and helps improve the performance and accuracy of the machine learning models.

## Normalization

Normalization is a preprocessing technique used to scale the features of the dataset so they have a similar range of values. 

The MinMax Scaler is a normalization technique that transforms features by scaling them to a specific range, typically between 0 and 1. This method preserves the relationships between the original data values and maintains the distribution of the data.

The Standard Scaler, also known as Z-score normalization, transforms the data such that the resulting distribution has a mean of 0 and a standard deviation of 1. This scaler is useful when the data follows a Gaussian distribution, and it helps in ensuring that each feature contributes equally to the model's performance.

## Model

 Multilayer Perceptron is used for developing a ML model. A MLP is composed of multiple layers of nodes, including an input layer, one or more hidden layers, and an output layer.

For robust evaluation, K-Fold Cross-Validation is used. This technique assesses the performance and generalizability of a model by dividing the dataset into K subsets, or folds, and then training and evaluating the model K times. In each iteration, one fold is used as the test set, while the remaining K-1 folds are used for training. This process is repeated K times, with each fold serving as the test set exactly once. The performance metrics from each iteration are then averaged to obtain a more reliable evaluation of the model's performance.

K-Fold Cross-Validation helps reduce the impact of dataset variability on model assessment, providing a more reliable estimate of how well a model is likely to perform on unseen data. It is a widely used technique in machine learning to ensure a comprehensive evaluation of a model's capabilities.

K-Fold Cross-Validation also aids in determining the optimal values of hyperparameters. The performance of different architectures is compared using the average metrics obtained through this cross-validation method. This variant of cross-validation improves upon the holdout method by ensuring that the model score does not depend on how the training and test sets are chosen. The final result is obtained by taking the arithmetic mean and standard deviation of the results from each iteration.

## Training

Different model settings are used, including variations in the number of hidden layers, the number of hidden units per layer, learning rate, number of epochs, and batch size. The criteria for evaluating the model include the mean squared error (MSE), the standard deviation of the MSE, training time, and the R-squared (R²) metric. These metrics help in assessing the accuracy, stability, and efficiency of the model under different configurations.



## Results
All the Y values range between -1 and 1. Despite being within this range, the maximum Y value and some other features are not consistent. Therefore, the results are obtained both with normalized Y values and without normalizing them for comparison.

As previously mentioned, two normalization methods are used in this work. Initially, these two methods are compared.

in the case of including y values in normalization: With the same model settings, the MSE loss for MinMax normalization was better than for Standard normalization(repeated with different model parameters).

In both normalization methods, the model performs better on the dataset where Y is not normalized. for example :

**Standard Normalization**

Model Settings: [512, 512, 512]

When Y is normalized:

Average MSE: 2.15×10−22.15×10−2

Average R²: 0.978

Average Time (s): 202.765

Standard Deviation of MSE: 3.26×10−33.26×10−3

When Y is not normalized:

Average MSE: 1.74×10−21.74×10−2

Average R²: 0.983

Average Time (s): 201.518

Standard Deviation of MSE: 1.58×10−31.58×10−3

**MinMax Normalization**

Model Settings: [128, 128, 128, 128, 64, 64, 64]

When Y is normalized:

Average MSE: 2.07×10−42.07×10−4

Average R²: 0.964

Average Time (s): 868.246

Standard Deviation of MSE: 7.56×10−57.56×10−5

When Y is not normalized:

Average MSE: 1.29×10−41.29×10−4

Average R²: 0.977

Average Time (s): 806.568

Standard Deviation of MSE: 1.34×10−51.34×10−5

Based on these results, MinMax normalization is chosen for evaluating hyperparameters.

