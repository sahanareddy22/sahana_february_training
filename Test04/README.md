# House Price Prediction using Supervised Machine Learning

## Problem Statement
The objective of this project is to predict house prices using supervised machine learning algorithms. The models learn patterns from housing data and estimate house prices based on various features.

## Dataset Description
The dataset contains housing information such as:

- Area
- Bedrooms
- Bathrooms
- Stories
- Parking
- Furnishing Status
- Price

Target variable: Price

## Data Cleaning Steps

1. Handling Missing Values
Missing values were replaced using median values.

2. Removing Duplicate Records
Duplicate rows were removed from the dataset.

3. Encoding Categorical Variables
Categorical features like furnishing status were converted into numerical form using One Hot Encoding.

4. Feature Scaling
StandardScaler was used to normalize numerical features.

5. Train Test Split
The dataset was split into 80% training and 20% testing data.

## Algorithms Used

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

## Evaluation Metrics

The models were evaluated using:

- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

## Results

Random Forest produced the best performance among the three models.

## Conclusion

Random Forest performed better because it combines multiple decision trees and reduces overfitting, giving more accurate predictions for house prices.