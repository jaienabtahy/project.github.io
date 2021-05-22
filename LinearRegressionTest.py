import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('FuelConsumption.csv.')
x = df[['ENGINESIZE']].values
y = df[['CO2EMISSIONS']].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
from LinearRegression import LinearRegression
regressor = LinearRegression(lr=0.001, n_iterations=1)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(predictions)
