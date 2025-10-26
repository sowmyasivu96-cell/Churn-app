from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("churn_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        age = int(request.form['Age'])
        plan = request.form['Plan']
        usage = int(request.form['Usage'])
        charges = float(request.form['MonthlyCharges'])

        # One-hot encode Plan manually
        plan_basic = 1 if plan == 'Basic' else 0
        plan_standard = 1 if plan == 'Standard' else 0
        plan_premium = 1 if plan == 'Premium' else 0

        # Make prediction
        features = [[age, usage, charges, plan_basic, plan_standard, plan_premium]]
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]  # Probability of churn

        return f"Churn Prediction: {'Yes' if prediction[0]==1 else 'No'}, Probability: {probability:.2f}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
