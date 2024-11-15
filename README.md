# House Price Prediction using XGBoost and Random Forest Regressors

This project uses machine learning models to predict house prices based on the California housing dataset. The models employed include XGBoost Regressor and Random Forest Regressor, with a performance comparison between the two models.

## Project Overview
The main objectives of this project are:
- Load and preprocess the California housing dataset.
- Visualize the dataset to understand correlations between features.
- Train an XGBoost Regressor and a Random Forest Regressor.
- Evaluate the models using metrics such as R-squared and Mean Absolute Error (MAE).
- Visualize the model predictions against actual prices.

## Dataset
The **California housing dataset** is fetched directly using `sklearn.datasets.fetch_california_housing()`. This dataset includes various features about California's housing and serves as the input for the regression models.

## Installation and Dependencies
Ensure you have Python installed along with the following packages:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

You can install the required packages using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Code Description
### Step 1: Import Necessary Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
```

### Step 2: Load the Dataset
```python
house_price_dataset = sklearn.datasets.fetch_california_housing()
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)
house_price_dataframe['price'] = house_price_dataset.target
```

### Step 3: Data Analysis and Visualization
```python
# Display the first few rows and check the dataset's structure
house_price_dataframe.head()
house_price_dataframe.describe()

# Correlation heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(house_price_dataframe.corr(), cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.show()
```

### Step 4: Data Splitting
```python
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```

### Step 5: Model Training and Evaluation
#### XGBoost Regressor
```python
model = XGBRegressor(n_estimators=100, subsample=0.8, max_depth=7, learning_rate=0.1, gamma=0, colsample_bytree=0.8)
model.fit(X_train, Y_train)

# Evaluate on training data
training_data_prediction = model.predict(X_train)
print('R Squared Error:', metrics.r2_score(Y_train, training_data_prediction))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_train, training_data_prediction))

# Visualize actual vs. predicted prices
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()

# Evaluate on test data
test_data_prediction = model.predict(X_test)
print('R Squared Error:', metrics.r2_score(Y_test, test_data_prediction))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, test_data_prediction))
```

#### Random Forest Regressor
```python
model1 = RandomForestRegressor()
model1.fit(X_train, Y_train)

# Evaluate on training data
training_data_prediction1 = model1.predict(X_train)
print('R Squared Error:', metrics.r2_score(Y_train, training_data_prediction1))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_train, training_data_prediction1))

# Visualize actual vs. predicted prices
plt.scatter(Y_train, training_data_prediction1)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()

# Evaluate on test data
test_data_prediction1 = model1.predict(X_test)
print('R Squared Error:', metrics.r2_score(Y_test, test_data_prediction1))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, test_data_prediction1))
```

## Results
- The **R Squared Error** and **Mean Absolute Error** are reported for both training and testing datasets.
- Model performance can be improved through hyperparameter tuning, feature engineering, and data preprocessing.

## Future Work
- Implement hyperparameter tuning using `GridSearchCV` for further optimization.
- Test with additional models or ensembling techniques.
- Apply feature scaling or normalization if needed.

## License
This project is open-source and available for use and modification.
