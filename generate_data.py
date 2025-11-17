import pandas as pd
import numpy as np

np.random.seed(42)

def generate_student_data(n=2700):
    status_probs = {
        "Summer course (Final < 25)": 0.25,
        "Summer course (low attendance)": 0.20,
        "Summer course (low mid/end)": 0.20,
        "Fail (total < 25)": 0.05,
        "Fail (total < 50)": 0.05,
        "FX  (retake opportunity)": 0.10,
        "Pass (after retake)": 0.10,
        "Pass": 0.04,
        "Scholarship": 0.01
    }

    statuses = list(status_probs.keys())
    probs = list(status_probs.values())

    data = []

    for _ in range(n):
        status = np.random.choice(statuses, p=probs)

        if status == "Scholarship":
            regMid = np.random.randint(90, 101)
            regEnd = np.random.randint(90, 101)
            Final = np.random.randint(90, 101)
            attendance = np.random.randint(95, 101)
        elif status == "Pass":
            regMid = np.random.randint(70, 90)
            regEnd = np.random.randint(70, 90)
            Final = np.random.randint(70, 90)
            attendance = np.random.randint(80, 101)
        elif status == "Pass (after retake)":
            regMid = np.random.randint(60, 70)
            regEnd = np.random.randint(60, 70)
            Final = np.random.randint(60, 70)
            attendance = np.random.randint(70, 100)
        elif status == "FX  (retake opportunity)":
            regMid = np.random.randint(50, 60)
            regEnd = np.random.randint(50, 60)
            Final = np.random.randint(50, 60)
            attendance = np.random.randint(60, 100)
        elif status == "Fail (total < 50)":
            regMid = np.random.randint(30, 50)
            regEnd = np.random.randint(30, 50)
            Final = np.random.randint(30, 45)
            attendance = np.random.randint(60, 100)
        elif status == "Fail (total < 25)":
            regMid = np.random.randint(0, 50)
            regEnd = np.random.randint(0, 50)
            Final = np.random.randint(0, 50)
            attendance = np.random.randint(40, 80)
        elif status == "Summer course (low mid/end)":
            regMid = np.random.randint(0, 50)
            regEnd = np.random.randint(0, 50)
            Final = np.random.randint(50, 80)
            attendance = np.random.randint(70, 100)
        elif status == "Summer course (low attendance)":
            regMid = np.random.randint(40, 100)
            regEnd = np.random.randint(40, 100)
            Final = np.random.randint(40, 100)
            attendance = np.random.randint(40, 59)
        elif status == "Summer course (Final < 25)":
            regMid = np.random.randint(0, 100)
            regEnd = np.random.randint(0, 100)
            Final = np.random.randint(0, 24)
            attendance = np.random.randint(50, 101)

        regMid = max(0, min(100, regMid + np.random.randint(-5, 6)))
        regEnd = max(0, min(100, regEnd + np.random.randint(-5, 6)))
        Final = max(0, min(100, Final + np.random.randint(-5, 6)))
        attendance = max(40, min(100, attendance + np.random.randint(-3, 4)))

        data.append([regMid, regEnd, Final, attendance, status])

    df = pd.DataFrame(data, columns=["regMid", "regEnd", "Final", "attendance", "status"])

    extra_cases = []

    for _ in range(20):
        extra_cases.append([
            np.random.randint(90, 101),
            np.random.randint(90, 101),
            np.random.randint(90, 101),
            np.random.randint(95, 101),
            "Scholarship"
        ])
    
    for _ in range(20):
        extra_cases.append([
            np.random.randint(30, 50),
            np.random.randint(30, 50),
            np.random.randint(30, 45),
            np.random.randint(60, 100),
            "Fail (total < 50)"
        ])

    for _ in range(20):
        extra_cases.append([
            np.random.randint(50, 60),
            np.random.randint(50, 60),
            np.random.randint(50, 60),
            np.random.randint(60, 100),
            "FX  (retake opportunity)"
        ])

    df_extra = pd.DataFrame(extra_cases, columns=df.columns)
    df = pd.concat([df, df_extra], ignore_index=True)

    df.to_csv("student_data_large.csv", index=False)
    print(f"student_data_large.csv created successfully with {len(df)} samples (extra rare cases added).")

if __name__ == "__main__":
    generate_student_data()
