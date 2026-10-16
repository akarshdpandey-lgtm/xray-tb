import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "tb_densenet121_best.pth")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="TBScan AI API", version="1.0.0")

# Restrict this list to your actual frontend domain after deployment if desired.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
class_to_idx = checkpoint.get("class_to_idx", {"no_tb": 0, "tb": 1})
if class_to_idx.get("no_tb") != 0 or class_to_idx.get("tb") != 1:
    raise RuntimeError(f"Unexpected class mapping: {class_to_idx}")

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.to(DEVICE)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

@app.get("/")
def root():
    return {"service": "TBScan AI API", "status": "ok", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "architecture": "DenseNet-121",
        "input_size": "224x224 RGB",
        "classes": ["no_tb", "tb"],
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a PNG or JPG image.")

    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 15 MB or smaller.")

    try:
        from io import BytesIO
        image = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()

    no_tb_score = float(probs[0])
    tb_score = float(probs[1])
    prediction = "TB" if tb_score >= no_tb_score else "No TB"

    return {
        "prediction": prediction,
        "tb_score": round(tb_score, 6),
        "no_tb_score": round(no_tb_score, 6),
        "threshold": 0.5,
        "note": "Experimental screening output only; not a clinical diagnosis."
    }
