from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the machine learning model (pre-trained and saved as model.pkl)
model = None
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Home route (loads the web page)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route (handles the prediction)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form['area']),
                    float(request.form['bedrooms']),
                    float(request.form['bathrooms']),
                    float(request.form['floors'])]
        final_features = [np.array(features)]
        
        # Make prediction using the loaded model
        prediction = model.predict(final_features)
        
        output = round(prediction[0], 2)
        
        return render_template('index.html', prediction_text=f'Estimated House Price: ${output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
