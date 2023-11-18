# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# import dataset
df = pd.read_csv("Startups.csv")
df.head()
df.info()
df.shape

# 
df = df.dropna()
df.isnull().sum().sum()
df['State'].value_counts()

def removestateoutlier(value):
    if value not in ['New York', 'California', 'Florida']:
        return 'Others'
    else:
        return value

# Convert categorical data into numeric value
df['State'] = df['State'].apply(removestateoutlier)
df['State'].value_counts()
df.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['State'] = le.fit_transform(df['State'])
df.head()

# Define Dependent(y) and Independent(y) value
x = df.drop(['Profit'], axis = 1)
y = df['Profit']

# Split the dataset for Training and Testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Import Regressor Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)

# Calculate Accuracy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
# Calculate R-squared (R2) score
r2 = r2_score(y_test, predictions)
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2) Score: {r2:.2f}')

# Create .sav file
pickle.dump(lr, open('./multiple.sav', 'wb'))