import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
    imputer_numerical = joblib.load(os.path.join(MODEL_DIR, 'imputer_numerical.pkl'))
    imputer_categorical = joblib.load(os.path.join(MODEL_DIR, 'imputer_categorical.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scale.pkl'))
    print("Models and preprocessors loaded successfully!")
except Exception as e:
    print(f"Error loading models or preprocessors: {e}")
    model, imputer_numerical, imputer_categorical, scaler = None, None, None, None

SCALED_FEATURES = [
    'temp', 'rain', 'snow',
    'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',
    'DayOfWeek', 'DayOfYear', 'WeekOfYear'
]

MODEL_COLUMNS = [
    'temp', 'rain', 'snow', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',
    'DayOfWeek', 'DayOfYear', 'WeekOfYear',
    'holiday_Columbus Day', 'holiday_Independence Day', 'holiday_Labor Day',
    'holiday_Martin Luther King Jr Day', 'holiday_Memorial Day', 'holiday_New Years Day',
    'holiday_No_Holiday', 'holiday_State Fair', 'holiday_Thanksgiving Day',
    'holiday_Veterans Day', 'holiday_Washingtons Birthday', 'weather_Clouds',
    'weather_Drizzle', 'weather_Fog', 'weather_Haze', 'weather_Mist', 'weather_Rain',
    'weather_Smoke', 'weather_Snow', 'weather_Squall', 'weather_Thunderstorm'
]

@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model not loaded. Please check server logs.")

    datetime_str = request.form['datetime_input']
    temp = float(request.form['temp'])
    rain = float(request.form['rain'])
    snow = float(request.form['snow'])
    weather = request.form['weather']
    holiday = request.form['holiday']

    input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)
    input_df['temp'] = temp
    input_df['rain'] = rain
    input_df['snow'] = snow

    dt_obj = pd.to_datetime(datetime_str)
    input_df['Year'] = dt_obj.year
    input_df['Month'] = dt_obj.month
    input_df['Day'] = dt_obj.day
    input_df['Hour'] = dt_obj.hour
    input_df['Minute'] = dt_obj.minute
    input_df['Second'] = dt_obj.second
    input_df['DayOfWeek'] = dt_obj.dayofweek
    input_df['DayOfYear'] = dt_obj.dayofyear
    input_df['WeekOfYear'] = dt_obj.isocalendar().week # CORRECTED LINE

    if f'holiday_{holiday}' in input_df.columns:
        input_df[f'holiday_{holiday}'] = True

    if f'weather_{weather}' in input_df.columns:
        input_df[f'weather_{weather}'] = True

    input_df = input_df[MODEL_COLUMNS]

    input_df_scaled = input_df.copy()
    input_df_scaled[SCALED_FEATURES] = scaler.transform(input_df_scaled[SCALED_FEATURES])

    prediction = model.predict(input_df_scaled)[0]
    prediction_text = f"Predicted Traffic Volume: {int(prediction):,}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)