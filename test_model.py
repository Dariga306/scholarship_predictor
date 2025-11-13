import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "rf_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

model = joblib.load(MODEL_PATH)

np.random.seed(42)

n_samples = 10
new_data = pd.DataFrame({
    "regMid": np.random.randint(0, 101, n_samples),
    "regEnd": np.random.randint(0, 101, n_samples),
    "Final": np.random.randint(0, 101, n_samples),
    "attendance": np.random.randint(50, 101, n_samples)
})

new_data["Total"] = 0.3 * new_data["regMid"] + 0.3 * new_data["regEnd"] + 0.4 * new_data["Final"]

preds = model.predict(new_data[["regMid", "regEnd", "Final", "attendance"]])
result_df = new_data.assign(Prediction=preds)

print("Random Predictions:\n")
print(result_df.to_string(index=False))
