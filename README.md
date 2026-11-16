# ☀️ SolarCheck API – Clasificador de paneles solares (ViT-Tiny)

Servicio web que recibe la foto de un panel fotovoltaico y devuelve su estado (6 clases) con la confianza del modelo.
Es el modelo de imágenes del proyecto final **SolarCheck IA** (electivo IA Generativa, UDD): lo llama automáticamente
un flujo de **n8n** que integra imagen, serie temporal de generación y bitácoras de mantención, y entrega una
recomendación generada por un LLM con aprobación humana.

🔗 **API publicada:** https://solarcheck-api.onrender.com
📊 **Tablero del proyecto:** https://datastudio.google.com/reporting/8bda4835-840e-4464-9457-10177b27dafd

---

## 1. Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| `GET` | `/` | Estado del servicio, nombre del modelo y sus métricas en test |
| `POST` | `/predict` | Recibe una imagen (`multipart/form-data`, campo `file`) y devuelve la clasificación |

Ejemplo:

```bash
curl -F "file=@panel.jpg" https://solarcheck-api.onrender.com/predict
```

```json
{
  "clase": "Dusty",
  "nombre": "Polvo acumulado",
  "confianza": 0.9939,
  "requiere_revision_humana": false,
  "probabilidades": {"Excremento de aves": 0.002, "Limpio": 0.003, "Polvo acumulado": 0.9939, "...": "..."},
  "modelo": "ViT-Tiny (balanced accuracy test 0.872)"
}
```

Si la confianza es menor a 60 %, `requiere_revision_humana` es `true`. Un archivo que no es imagen devuelve error 400.

## 2. Modelo

- **Arquitectura:** ViT-Tiny (Vision Transformer, ~5,5 M parámetros, preentrenado en ImageNet con `timm`), ajustado a 6 clases.
- **Clases:** Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered.
- **Formato:** exportado de PyTorch a **ONNX** (diferencia máxima con PyTorch ≈ 1e-6) para correr en un servidor gratuito con ~180 MB de RAM.
- **Normalización:** se lee desde `clases.json` (media y desviación propias del modelo).

### Resultados (mismo conjunto de test, 131 imágenes)

| Modelo | Bal. accuracy validación | Accuracy test | Bal. accuracy test | Recall daños (eléctrico + físico) |
|---|---|---|---|---|
| Inspección por umbral de color/bordes (alternativa simple) | – | 0,351 | 0,329 | 0,10 |
| EfficientNet-B0 (CNN) | 0,853 | 0,809 | 0,815 | 0,817 |
| **ViT-Tiny (elegido)** | **0,916** | **0,870** | **0,872** | **0,867** |

El modelo se eligió por **balanced accuracy en validación**; el test solo se usó para reportar. Ambos modelos se entrenaron con la misma receta: 15 épocas, AdamW, *data augmentation* y pesos por clase por el desbalance.

## 3. Datos

| Campo | Detalle |
|---|---|
| Dataset | *Solar Panel Images – Clean and Faulty Images* (Afroz, Kaggle `pythonafroz/solar-panel-images`) |
| Licencia | No especificada; uso exclusivamente académico |
| Imágenes | 869 en 6 clases (se excluyó la clase `New`, con solo 16 imágenes) |
| División | 70 / 15 / 15 estratificada, semilla 42, **rutas ordenadas antes de dividir** |
| Reproducibilidad | `division.csv` guarda la partición de cada imagen; se verificó que no haya imágenes repetidas entre particiones |

**Limitaciones:** una sola fuente de fotos, clases desbalanceadas (69 fotos de daño físico frente a 193 de panel limpio), test pequeño en las clases de daño, y fotos heterogéneas en ángulo e iluminación. No reemplaza la inspección de personal calificado.

## 4. Archivos

```text
├── app.py            # API FastAPI: carga el modelo ONNX y expone / y /predict
├── modelo.onnx       # ViT-Tiny exportado a ONNX
├── clases.json       # clases, nombre del modelo, métricas de test y normalización
├── division.csv      # partición train/val/test de cada imagen
├── requirements.txt  # fastapi, uvicorn, python-multipart, pillow, numpy, onnxruntime
└── README.md
```

## 5. Ejecutar localmente

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
# abrir http://localhost:8000
```

## 6. Despliegue en Render

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Plan gratuito: el servicio se suspende tras ~15 minutos sin uso y la primera consulta tarda hasta ~50 s. El flujo de n8n reintenta 5 veces y, si no hay respuesta, avisa que corresponde inspección manual.

## 7. Integración en el proyecto

```text
Foto + leyenda "INV-xx" (Telegram)
  → esta API (ViT-Tiny) → validación en Python (confianza, margen, prioridad)
  → generación del inversor (serie temporal) + bitácoras (Transformer de texto)
  → LLM Gemini: recomendación integrada en JSON, con citas verificadas
  → Telegram + Gmail + historial en Google Sheets + alerta
  → aprobación del jefe por Telegram → orden de trabajo
```

## 8. Integrantes

- Benjamín Araya
- Guillermo Barañao
- Fernando Rodríguez

## 9. Uso de IA

Se usaron asistentes de IA (Claude, ChatGPT) para generar y revisar código, depurar errores y redactar documentación.
El grupo revisó y puede explicar cada parte del sistema. El LLM del flujo (Google Gemini) solo recibe resultados
textuales de los modelos, nunca la fotografía. Ninguna clave de API está en este repositorio.
