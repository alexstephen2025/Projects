from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("delhi_gurugram_price_model.keras")
scaler = joblib.load("scaler.pkl")

# Load and prep column template
raw_data = pd.read_csv("delhi_gurugram_real_estate_dataset.csv")
dummy_df = pd.get_dummies(raw_data.drop("Price_INR", axis=1), drop_first=True)
expected_columns = pd.read_csv("trained_columns.csv").columns

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    age = int(request.form['age'])
    location = request.form['location']
    property_type = request.form['property_type']
    city = request.form['city']

    # Build the input dictionary with one-hot encoding for categorical variables
    input_data = {
        'Area_sqft': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age_of_Property': age,
        f'Property_Type_{property_type}': 1,
        f'Location_{location}': 1,
        f'City_{city}': 1
    }

    # Create DataFrame from input data
    input_df = pd.DataFrame([input_data])

    # Ensure all expected columns are present, with 0s for missing ones
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Scale input data using the saved scaler
    scaled_input = scaler.transform(input_df)

    # Predict the price using the model
    prediction = model.predict(scaled_input)[0][0]
    price = round(prediction, 2)

    return render_template('result.html', price=price)

if __name__ == '__main__':
    app.run(debug=True)
