# SolarCheck API

Servicio que clasifica fotos de paneles solares con EfficientNet-B0 (6 clases), exportado a ONNX.
Lo llama el flujo de n8n del proyecto final de IA Generativa (UDD).

- `GET /` → estado del servicio
- `POST /predict` → enviar la foto en el campo `file` (multipart/form-data); devuelve clase, confianza y probabilidades.

Despliegue en Render:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
