import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# ---------------------------------------------------------
# Dataset 1: Car Details (Predicting Selling Price)
# ---------------------------------------------------------
try:
    df_cars = pd.read_csv(r"C:\Users\Haider Ali\Downloads\CAR DETAILS FROM CAR DEKHO.csv")
    X_cars = df_cars[['year', 'km_driven']]
    y_cars = df_cars['selling_price']
    
    model_cars = LinearRegression()
    model_cars.fit(X_cars, y_cars)
    joblib.dump(model_cars, 'car_model.pkl')
    print("Car model saved successfully as 'car_model.pkl'")
except FileNotFoundError:
    print("Error: Car dataset not found at the specified path.")

# ---------------------------------------------------------
# Dataset 2: Energy Data (Predicting Energy Consumption)
# ---------------------------------------------------------
try:
    df_energy = pd.read_csv(r"C:\Users\Haider Ali\Downloads\test_energy_data.csv")
    X_energy = df_energy[['Square Footage', 'Number of Occupants', 'Appliances Used']]
    y_energy = df_energy['Energy Consumption']
    
    model_energy = LinearRegression()
    model_energy.fit(X_energy, y_energy)
    joblib.dump(model_energy, 'energy_model.pkl')
    print("Energy model saved successfully as 'energy_model.pkl'")
except FileNotFoundError:
    print("Error: Energy dataset not found at the specified path.")

print("All training complete.")