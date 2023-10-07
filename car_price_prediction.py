import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
try:
    data = pd.read_csv('/kaggle/input/cars-dataset-audi-bmw-ford-hyundai-skoda-vw/cars_dataset.csv')
except FileNotFoundError:
    data = pd.read_csv('cars_dataset.csv')

# Perform exploratory data analysis (optional)
print("Basic info about the dataset:")
print(data.info())

print("\nSummary statistics of the dataset:")
print(data.describe())

print("\nMissing values in the dataset:")
print(data.isnull().sum())

print("\nColumns in the dataset:")
print(data.columns)

# Select features (independent variables) and target variable (car price)
X = data[['year', 'mileage', 'tax', 'mpg', 'engineSize']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

# Plot actual vs. predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Car Prices")
plt.show()

# Predict the price of a new car
new_car_features = np.array([[2019, 15000, 145, 50, 2.0]])
predicted_price = model.predict(new_car_features)
print("\nPredicted Price for a New Car:", predicted_price[0])
