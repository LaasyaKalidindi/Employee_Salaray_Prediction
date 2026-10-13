
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("salary_model.pkl")


def format_indian_salary(salary):
    salary = float(salary)

    if salary >= 1_00_00_000:
        return f"₹{salary / 1_00_00_000:.2f} crore per year"

    elif salary >= 1_00_000:
        return f"₹{salary / 1_00_000:.2f} lakh per year"

    else:
        amount = f"{salary:,.0f}"
        return f"₹{amount} per year"


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        employee = pd.DataFrame([{
            "Experience": float(request.form["experience"]),
            "Education": request.form["education"],
            "JobRole": request.form["job_role"],
            "Location": request.form["location"]
        }])

        salary = model.predict(employee)[0]
        prediction = format_indian_salary(salary)

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)
