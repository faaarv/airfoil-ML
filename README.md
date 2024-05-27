# airfoil-ML
## Introduction

In this repository,  Airfoil's Lift and Drag Coefficient is predicted using Machine Learning

## Dataset
 The dataset used in this work is available at: https://github.com/nasa/airfoil-learning.

This dataset ( not normalized ) contains the geometry (x, y) of airfoils and the XFOIL results (Cd, Cdp, Cm , Cl , Cp for upper and lower surface) for various Reynolds numbers (Re), angles of attack (AoA), and Ncrit for each airfoil. 



after  removing unnecessary fields such as  Cp, Cdp, Cm, x. A dataframe is created from original dataset. The x values are the same for all the  airfoils so they are removed from the features. then the dataset is normalized using two scalers: MinMax and Standard.


Preprocessing is a crucial step in machine learning, that involves transforming raw data into a format suitable for modeling. It enhances the quality of the data and helps improve the performance and accuracy of the machine learning models.
