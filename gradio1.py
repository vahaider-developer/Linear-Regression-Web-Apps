import gradio as gr
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

model_cars = joblib.load('car_model.pkl')
model_energy = joblib.load('energy_model.pkl')

def predict_car(year, km_driven):
    pred = model_cars.predict([[year, km_driven]])
    return f"Predicted Selling Price: ₹{pred[0]:,.2f}"

def predict_energy(sq_ft, occupants, appliances):
    pred = model_energy.predict([[sq_ft, occupants, appliances]])
    return f"Predicted Energy Consumption: {pred[0]:.2f} units"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Dual Dataset Linear Regression Predictor")
    
    with gr.Tab("Car Price Predictor"):
        with gr.Row():
            year = gr.Number(label="Manufacturing Year (e.g., 2015)")
            km_driven = gr.Number(label="Kilometers Driven")
        btn_car = gr.Button("Predict Car Price")
        out_car = gr.Textbox(label="Prediction Result")
        btn_car.click(predict_car, inputs=[year, km_driven], outputs=out_car)

    with gr.Tab("Energy Consumption Predictor"):
        with gr.Row():
            sq_ft = gr.Number(label="Square Footage")
            occupants = gr.Number(label="Number of Occupants")
            appliances = gr.Number(label="Appliances Used")
        btn_energy = gr.Button("Predict Energy")
        out_energy = gr.Textbox(label="Prediction Result")
        btn_energy.click(predict_energy, inputs=[sq_ft, occupants, appliances], outputs=out_energy)

if __name__ == "__main__":
    demo.launch()