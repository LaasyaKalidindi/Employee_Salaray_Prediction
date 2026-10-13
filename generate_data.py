
import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

experience = np.random.uniform(0, 25, n)

education = np.random.choice(
    ["Bachelor", "Master", "PhD"],
    n
)

job_role = np.random.choice(
    ["Software Engineer", "Data Analyst",
     "Data Scientist", "Manager"],
    n
)

location = np.random.choice(
    ["Hyderabad", "Bangalore", "Mumbai", "Delhi"],
    n
)

education_bonus = {
    "Bachelor": 0,
    "Master": 180000,
    "PhD": 350000
}

role_bonus = {
    "Software Engineer": 200000,
    "Data Analyst": 100000,
    "Data Scientist": 350000,
    "Manager": 400000
}

location_bonus = {
    "Hyderabad": 0,
    "Bangalore": 150000,
    "Mumbai": 120000,
    "Delhi": 80000
}

salary = (
    400000
    + experience * 70000
    + np.array([education_bonus[x] for x in education])
    + np.array([role_bonus[x] for x in job_role])
    + np.array([location_bonus[x] for x in location])
    + np.random.normal(0, 100000, n)
)

df = pd.DataFrame({
    "Experience": experience.round(1),
    "Education": education,
    "JobRole": job_role,
    "Location": location,
    "Salary": salary.round(0).astype(int)
})

df.to_csv("employee_data.csv", index=False)

print(df.head())
print("Dataset created successfully!")
