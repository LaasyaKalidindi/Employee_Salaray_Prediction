
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("employee_data.csv")
df = df.drop_duplicates().dropna()

# Exploratory Data Analysis
sns.barplot(data=df, x="JobRole", y="Salary")
plt.title("Average Salary by Job Role")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("salary_by_role.png")
plt.close()

sns.scatterplot(data=df, x="Experience", y="Salary")
plt.title("Experience vs Salary")
plt.tight_layout()
plt.savefig("experience_vs_salary.png")
plt.close()

# Features and target
X = df[["Experience", "Education", "JobRole", "Location"]]
y = df["Salary"]

numeric_features = ["Experience"]
categorical_features = ["Education", "JobRole", "Location"]

preprocessor = ColumnTransformer([
    ("categorical",
     OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
], remainder="passthrough")

# Split data before fitting the preprocessing pipeline
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    )
}

results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results[name] = {
        "pipeline": pipeline,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print(f"\n{name}")
    print(f"MAE:  ₹{mae:,.0f}")
    print(f"RMSE: ₹{rmse:,.0f}")
    print(f"R²:   {r2:.3f}")

# Select the model with the lowest MAE
best_name = min(results, key=lambda name: results[name]["MAE"])
best_model = results[best_name]["pipeline"]

joblib.dump(best_model, "salary_model.pkl")

print(f"\nBest model: {best_name}")
print("Saved as salary_model.pkl")
