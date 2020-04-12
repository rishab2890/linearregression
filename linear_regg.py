import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataframe = pd.read_csv('challenge_dataset.csv')
x_values = dataframe[['x']]
y_values = dataframe[['y']]

model = LinearRegression()
model.fit(x_values, y_values)

plt.scatter(x_values, y_values)
plt.plot(x_values, model.predict(x_values))
plt.show()
