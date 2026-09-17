from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the diabetes model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None

    if request.method == "POST":

        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])
        skin_thickness = float(request.form["skin_thickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        pedigree = float(request.form["pedigree"])
        age = float(request.form["age"])

        # Your trained model was created without Insulin.
        # Therefore Insulin is collected by the UI but not passed to the model.

        features = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            bmi,
            pedigree,
            age
        ]])

        # Apply the same scaler used during training
        scaled_features = scaler.transform(features)

        # Make prediction
        result = model.predict(scaled_features)[0]

        # Get probability
        probability = model.predict_proba(scaled_features)[0][1] * 100

        if result == 1:
            prediction = "Diabetes detected"
        else:
            prediction = "No diabetes detected"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)