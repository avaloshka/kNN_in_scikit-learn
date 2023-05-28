import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

import numpy as np

url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
)

abalone = pd.read_csv(url)
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shoucked weight', 'Viscera weight', 'Shell weight', 'Rings']
# get rid of strings (column "Sex") to evaluate just numbers, ofcourse
abalone = abalone.drop('Sex', axis=1)

# We will try to predict "Rings" so lets drop the column to train a set
X = abalone.drop("Rings", axis=1)
# Convert DataFrame to numpy array
X = X.values

# will use "Rings" as a target value
y = abalone["Rings"]
y = y.values

# we want to predict number of Rings

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

# Create kNN model
knn_model = KNeighborsRegressor(n_neighbors=3)

# train model
knn_model.fit(X_train, y_train)

# make predictions
pred_train = knn_model.predict(X_train)
pred_test = knn_model.predict(X_test)

mse_train = mean_squared_error(y_train, pred_train)
rmse_train = np.sqrt(mse_train)

mse_test = mean_squared_error(y_test, pred_test)
rmse_test = np.sqrt(mse_test)

# new model
knn_model_25 = KNeighborsRegressor(n_neighbors=25)
knn_model_25.fit(X_train, y_train)
pred_test_25 = knn_model_25.predict(X_test)
mse_test_25 = mean_squared_error(y_test, pred_test_25)
rmse_test_25 = np.sqrt(mse_test_25)
print(rmse_test_25)