import gradio as gr
import pandas as pd
import joblib

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

def predict_price(
    MedInc, HouseAge, AveRooms, AveBedrms,
    Population, AveOccup, Latitude, Longitude
):
    input_df = pd.DataFrame([[
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]], columns=FEATURES)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    return f"${prediction * 100000:,.2f}"

demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(0.5, 15, 0.1, label="Median Income"),
        gr.Slider(1, 60, 1, label="House Age"),
        gr.Slider(1, 10, 0.1, label="Average Rooms"),
        gr.Slider(1, 5, 0.1, label="Average Bedrooms"),
        gr.Slider(100, 5000, 50, label="Population"),
        gr.Slider(1, 10, 0.1, label="Average Occupancy"),
        gr.Slider(32, 42, 0.1, label="Latitude"),
        gr.Slider(-125, -114, 0.1, label="Longitude"),
    ],
    outputs=gr.Textbox(label="Predicted House Price"),
    title="🏠 House Price Prediction (Linear Regression)",
    description="Linear Regression model trained on California Housing dataset."
)

demo.launch()
