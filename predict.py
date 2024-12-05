# predict.py

import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


area = 2000          
bedrooms = 4        
bathrooms = 3       
floors = 2           

# Create a DataFrame for the input data
sample_data = pd.DataFrame([[area, bedrooms, bathrooms, floors]], columns=['area', 'bedrooms', 'bathrooms', 'floors'])

# Make prediction
prediction = loaded_model.predict(sample_data)

# Display the predicted house price
print(f"Predicted house price: ${prediction[0]:,.2f}")
