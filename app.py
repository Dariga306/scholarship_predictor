from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib, os

app = Flask(__name__)

def clamp_score(x):
    try:
        x = int(float(x)) 
    except Exception:
        return None
    if x < 0: return 0
    if x > 100: return 100
    return x


WEIGHTS = {'mid': 0.30, 'end': 0.30, 'final': 0.40}
MIN_ATTENDANCE = 70
MIN_MID_END = 25
FX_LOWER = 25
FX_UPPER = 50
PASS_THRESHOLD = 50
SCHOLAR_THRESHOLD = 70

def academic_outcome_safe(regMid, regEnd, Final, attendance, fx_score=None):
    regMid = clamp_score(regMid)
    regEnd = clamp_score(regEnd)
    Final = clamp_score(Final)
    attendance = clamp_score(attendance)
    fx_score = clamp_score(fx_score) if fx_score is not None else None

    if None in (regMid, regEnd, Final, attendance):
        return {'status': 'ERROR', 'category': 'Invalid', 'reason': 'Invalid input values', 'total': None}

    if attendance < MIN_ATTENDANCE:
        return {'status': 'FAIL', 'category': 'Summer course ', 'reason': 'low attendance', 'total': None}

    if regMid < MIN_MID_END or regEnd < MIN_MID_END:
        return {'status': 'FAIL', 'category': 'Summer course ', 'reason': 'low mid/end', 'total': None}

    if Final < FX_LOWER:
        return {'status': 'FAIL', 'category': 'Summer course ', 'reason': 'Final < 25', 'total': None}

    def total_with(score):
        return regMid * WEIGHTS['mid'] + regEnd * WEIGHTS['end'] + score * WEIGHTS['final']

    if 45 <= Final < 50:
        if fx_score is None:
            return {'status': 'RETAKE', 'category': 'FX ', 'reason': 'retake required', 'total': None}
        if fx_score < FX_UPPER:
            return {'status': 'FAIL', 'category': 'Summer course ', 'reason': 'failed retake (FX < 50)', 'total': None}
        total = round(total_with(fx_score), 2)
        if total >= SCHOLAR_THRESHOLD:
            return {'status': 'PASS', 'category': 'Scholarship ', 'reason': 'after retake', 'total': total}
        return {'status': 'PASS', 'category': 'Pass ', 'reason': 'after retake', 'total': total}

    total = round(total_with(Final), 2)
    if total < PASS_THRESHOLD:
        return {'status': 'FAIL', 'category': 'Fail ', 'reason': 'total < 50', 'total': total}
    if total < SCHOLAR_THRESHOLD:
        return {'status': 'PASS', 'category': 'Pass ', 'reason': 'no scholarship', 'total': total}
    return {'status': 'PASS', 'category': 'Scholarship ', 'reason': 'scholarship', 'total': total}

MODEL_PATH = "rf_model.pkl"
DATA_PATH = "student_data_large.csv"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    data = pd.read_csv(DATA_PATH)
    X = data[['regMid', 'regEnd', 'Final', 'attendance']]
    y = data['result']
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/after_final", methods=["POST"])
def after_final():
    data = request.get_json()
    regMid = data.get("regMid")
    regEnd = data.get("regEnd")
    final = data.get("Final")
    attendance = data.get("attendance")
    fx = data.get("fx_score")

    result = academic_outcome_safe(regMid, regEnd, final, attendance, fx)
    return jsonify({"academic_result": result["category"], "details": result})

@app.route("/api/before_final", methods=["POST"])
def before_final():
    data = request.get_json()
    regMid = clamp_score(data.get("regMid"))
    regEnd = clamp_score(data.get("regEnd"))
    attendance = clamp_score(data.get("attendance"))

    if None in (regMid, regEnd, attendance):
        return jsonify({"error": "Invalid input"}), 400

    best_final = round((70 - (regMid * 0.3 + regEnd * 0.3)) / 0.4, 2)
    pass_final = round((50 - (regMid * 0.3 + regEnd * 0.3)) / 0.4, 2)

    return jsonify({
        "needed_for_pass": min(max(pass_final, 0), 100),
        "needed_for_scholarship": min(max(best_final, 0), 100)
    })

if __name__ == "__main__":
    app.run(debug=True)
