"""Flask web application for the student performance estimate demo."""
from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_model.joblib"
INFO_PATH = BASE_DIR / "models" / "model_info.json"
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

if not MODEL_PATH.exists() or not INFO_PATH.exists():
    raise RuntimeError("Model artifacts are missing. Run `python train_model.py` first.")
model = joblib.load(MODEL_PATH)
model_info = json.loads(INFO_PATH.read_text(encoding="utf-8"))

REQUIRED = {
    "study_hours": (0, 24), "attendance": (0, 100), "previous_score": (0, 100),
    "assignment_completion": (0, 100), "sleep_hours": (0, 24),
}


def category(score):
    if score >= 85: return "Excellent"
    if score >= 70: return "Good"
    if score >= 50: return "Average"
    return "Needs Improvement"


def suggestions(values):
    tips = []
    if values["study_hours"] < 2: tips.append("Build a steady daily study routine, starting with focused 25-minute sessions.")
    elif values["study_hours"] < 4: tips.append("Try adding one focused revision block to your current study routine.")
    if values["attendance"] < 75: tips.append("Attend classes regularly and review notes for any missed lessons.")
    if values["previous_score"] < 60: tips.append("Revisit topics from the previous exam and ask for help on difficult concepts.")
    if values["assignment_completion"] < 80: tips.append("Complete assignments on time; they are useful practice for exam questions.")
    if values["sleep_hours"] < 7: tips.append("Aim for a consistent, adequate sleep schedule to support learning and recall.")
    if values["internet_access"] == "No": tips.append("Use library or downloaded learning resources when online access is unavailable.")
    if values["extracurricular"] == "No": tips.append("Balance study with a healthy activity or short breaks to support wellbeing.")
    if not tips: tips.append("Keep your balanced habits, review regularly, and use practice questions to check understanding.")
    return tips


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/predict")
def predict_page():
    return render_template("predict.html")


@app.get("/dashboard")
def dashboard():
    return render_template("dashboard.html", info=model_info)


@app.get("/about")
def about():
    return render_template("about.html")


@app.get("/api/model-info")
def api_model_info():
    return jsonify(model_info)


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send prediction fields as a JSON object."}), 400
    clean = {}
    errors = []
    for field, (low, high) in REQUIRED.items():
        try:
            value = float(payload.get(field, ""))
            if not (low <= value <= high): raise ValueError
            clean[field] = value
        except (TypeError, ValueError):
            errors.append(f"{field.replace('_', ' ').capitalize()} must be a number from {low} to {high}.")
    for field in ("internet_access", "extracurricular"):
        value = payload.get(field)
        if value not in ("Yes", "No"):
            errors.append(f"{field.replace('_', ' ').capitalize()} must be Yes or No.")
        else: clean[field] = value
    if errors:
        return jsonify({"error": "Please correct the highlighted inputs.", "details": errors}), 400
    clean["student_name"] = str(payload.get("student_name", "")).strip()[:80]
    row = pd.DataFrame([{field: clean[field] for field in model_info["feature_names"]}])
    estimate = float(model.predict(row)[0])
    estimate = max(0.0, min(100.0, estimate))
    return jsonify({"student_name": clean["student_name"], "predicted_score": round(estimate, 1),
                    "category": category(estimate), "explanation": "This estimate comes from patterns learned from the project's synthetic training data and the values you entered. It is not a guaranteed exam result.",
                    "suggestions": suggestions(clean)})


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"error": "The server could not complete this request. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True)
