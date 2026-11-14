"""SolarCheck API: servicio que clasifica fotos de paneles solares.

n8n envía una foto (POST /predict) y recibe un JSON con la clase,
la confianza y las probabilidades del modelo EfficientNet-B0 entrenado
en el Certamen 1. El modelo se exportó de PyTorch a ONNX para que el
servicio sea liviano y quepa en un servidor gratuito (512 MB de RAM).
"""
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parent
UMBRAL_CONFIANZA = 0.60  # bajo este valor se pide revisión humana

NOMBRES = {
    "Bird-drop": "Excremento de aves",
    "Clean": "Limpio",
    "Dusty": "Polvo acumulado",
    "Electrical-damage": "Daño eléctrico",
    "Physical-Damage": "Daño físico",
    "Snow-Covered": "Cubierto de nieve",
}

# Carga del modelo una sola vez, al iniciar el servicio.
# clases.json (generado en Colab) trae las clases, el nombre del modelo, sus métricas en test
# y la normalización propia del modelo (ImageNet para EfficientNet, 0.5 para ViT).
with open(ROOT / "clases.json", encoding="utf-8") as f:
    META = json.load(f)
CLASES = META["clases"]
MEDIA = np.array(META.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
DESV = np.array(META.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
NOMBRE_MODELO = META.get("modelo", "EfficientNet-B0")
SESION = ort.InferenceSession(str(ROOT / "modelo.onnx"))

app = FastAPI(title="SolarCheck API")


def preprocesar(img: Image.Image) -> np.ndarray:
    """Redimensiona a 224x224, escala a [0,1], normaliza y ordena como (1, 3, 224, 224)."""
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEDIA) / DESV
    return x.transpose(2, 0, 1)[np.newaxis, :]


def softmax(z: np.ndarray) -> np.ndarray:
    """Convierte las salidas del modelo (logits) en probabilidades que suman 1."""
    e = np.exp(z - z.max())
    return e / e.sum()


@app.get("/")
def estado():
    """Sirve para comprobar que el servicio está encendido."""
    return {"estado": "ok", "modelo": NOMBRE_MODELO, "test_accuracy": META.get("test_accuracy"),
            "test_balanced_accuracy": META.get("test_balanced_accuracy"), "clases": CLASES}


@app.post("/predict")
async def predecir(file: UploadFile = File(...)):
    """Recibe una imagen y devuelve la clasificación del panel."""
    contenido = await file.read()
    try:
        img = Image.open(BytesIO(contenido))
    except Exception:
        raise HTTPException(status_code=400, detail="El archivo enviado no es una imagen válida")

    logits = SESION.run(None, {"input": preprocesar(img)})[0][0]
    probs = softmax(logits)
    i_max = int(probs.argmax())
    clase = CLASES[i_max]
    confianza = round(float(probs[i_max]), 4)

    return {
        "clase": clase,
        "nombre": NOMBRES.get(clase, clase),
        "confianza": confianza,
        "requiere_revision_humana": confianza < UMBRAL_CONFIANZA,
        "probabilidades": {NOMBRES.get(c, c): round(float(p), 4) for c, p in zip(CLASES, probs)},
        "modelo": f"{NOMBRE_MODELO} (balanced accuracy test {META.get('test_balanced_accuracy')})",
    }
