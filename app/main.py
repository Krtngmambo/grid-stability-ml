import joblib, numpy as np
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("model/grid_stability_rf.joblib")
scaler = joblib.load("model/scaler.joblib")

app = FastAPI(title="Grid Stability Prediction API", version="1.0.0")
FEATURES = ["tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4"]

class GridFeatures(BaseModel):
    tau1: float; tau2: float; tau3: float; tau4: float
    p1: float; p2: float; p3: float; p4: float
    g1: float; g2: float; g3: float; g4: float

def run_model(values):
    xs = scaler.transform(np.array([values], dtype=float))
    pred = int(model.predict(xs)[0])
    conf = float(model.predict_proba(xs)[0].max())
    return ("stable" if pred == 1 else "unstable"), round(conf, 4)

# ----- API (untuk curl / Postman) -----
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(f: GridFeatures):
    label, conf = run_model([getattr(f, k) for k in FEATURES])
    return {"prediction": label, "confidence": conf}

# ----- UI Interaktif (Gradio) -----
def ui_predict(*vals):
    label, conf = run_model(list(vals))
    return f"Prediksi: {label.upper()}  |  Confidence: {conf}"

inputs = [gr.Number(label=name, value=0.0) for name in FEATURES]
demo = gr.Interface(
    fn=ui_predict, inputs=inputs, outputs=gr.Text(label="Hasil"),
    title="⚡ Prediksi Stabilitas Jaringan Listrik",
    description="Isi 12 parameter lalu klik Submit.")

app = gr.mount_gradio_app(app, demo, path="/ui")