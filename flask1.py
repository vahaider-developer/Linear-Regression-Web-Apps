from flask import Flask, request, render_template
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

model_cars = joblib.load('car_model.pkl')
model_energy = joblib.load('energy_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_car', methods=['POST'])
def predict_car():
    year = float(request.form['year'])
    km_driven = float(request.form['km_driven'])
    
    pred = model_cars.predict([[year, km_driven]])
    result = f"₹{pred[0]:,.2f}"
    
    return render_template('index.html', result_car=result)

@app.route('/predict_energy', methods=['POST'])
def predict_energy():
    sq_ft = float(request.form['sq_ft'])
    occupants = float(request.form['occupants'])
    appliances = float(request.form['appliances'])
    
    pred = model_energy.predict([[sq_ft, occupants, appliances]])
    result = f"{pred[0]:.2f} units"
    
    return render_template('index.html', result_energy=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)