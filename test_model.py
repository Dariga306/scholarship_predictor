import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "rf_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

model = joblib.load(MODEL_PATH)
print("Model loaded successfully.\n")

n_samples = 10
new_data = pd.DataFrame({
    "regMid": np.random.randint(0, 101, n_samples),
    "regEnd": np.random.randint(0, 101, n_samples),
    "Final": np.random.randint(0, 101, n_samples),
    "attendance": np.random.randint(50, 101, n_samples)
})

preds = model.predict(new_data)
result_df = new_data.assign(Prediction=preds)

print("Random Predictions:\n")
print(result_df.to_string(index=False))
