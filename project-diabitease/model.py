import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load & clean dataset
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    replace_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in replace_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    return df

# Train model
def train_model():
    df = load_data()
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # Choose RF (more stable)
    return rf, scaler

model, scaler = train_model()

def predict(values):
    arr = np.array(values).reshape(1, -1)
    # No scaling needed for RF
    proba = model.predict_proba(arr)[0][1]
    prediction = int(proba >= 0.5)
    return prediction, round(proba, 3)
