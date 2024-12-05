# Example model training script (train_model.py)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load your dataset (replace with your dataset path)
data = pd.read_csv('house_data.csv')

# Select features and the target variable
X = data[['area', 'bedrooms', 'bathrooms', 'floors']]  
y = data['price'] 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model as a .pkl file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
