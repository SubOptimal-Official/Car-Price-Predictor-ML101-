from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load ML model and dataset
model = pickle.load(open('CarPrice_Predictor/LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv("CarPrice_Predictor/clean_car.csv")

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique()) 
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())

    return render_template('index.html',
                           companies=companies,
                           car_models=car_models,
                           years=year,
                           fuel_types=fuel_type)

# ✅ ADD this missing route
@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    driven = int(request.form.get('kilo_driven'))

    # Make sure feature order matches model input
    input_df = pd.DataFrame([[car_model, company, year, driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    prediction = model.predict(input_df)

    return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)
