from flask import Flask, render_template, request
from model import predict

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        vals = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DPF"]),
            float(request.form["Age"])
        ]
    except:
        return "<h2>Invalid input!</h2>"

    pred, proba = predict(vals)

    return render_template(
        "result.html",
        pred=pred,
        proba=proba,
        vals=vals
    )

if __name__ == "__main__":
    app.run(debug=True)
