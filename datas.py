from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("churn_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        age = int(request.form['Age'])
        plan = request.form['Plan']
        usage = int(request.form['Usage'])
        charges = float(request.form['MonthlyCharges'])
        # Convert plan to one-hot, etc.
        prediction = model.predict([[age, usage, charges, plan_basic, plan_standard, plan_premium]])
        return f"Churn Probability: {prediction[0]}"
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
