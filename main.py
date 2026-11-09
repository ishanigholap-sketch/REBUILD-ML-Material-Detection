
import os
import random
import json
from functools import lru_cache

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from material_analysis import analyze_materials


# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))

CLASS_MAP = {
    "concrete": "concrete",
    "brick": "brick",
    "steel": "steel",
    "wood": "wood",
    "glass": "glass",
    "tiles": "tiles",
}

VALID_MATERIALS = list(CLASS_MAP.values())


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="RE:BUILD Material Detection API",
    description=(
        "Construction material detection, "
        "material analysis and recovery prediction"
    ),
    version="1.3.0"
)

origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# LOAD YOLO MODEL
# ==========================================

@lru_cache(maxsize=1)
def get_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}"
        )

    return YOLO(MODEL_PATH)


# ==========================================
# IMAGE PROCESSING
# ==========================================

async def read_image(file: UploadFile):

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Only JPG, PNG and WEBP images are allowed."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(contents) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds maximum allowed size."
        )

    image_array = np.frombuffer(
        contents,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    height, width = image.shape[:2]

    return image, width, height


# ==========================================
# MOCK DETECTION
# ==========================================

def mock_detection(width, height):

    selected = random.sample(
        VALID_MATERIALS,
        k=random.randint(2, 4)
    )

    detections = []

    for material in selected:

        x1 = random.randint(0, width // 2)
        y1 = random.randint(0, height // 2)

        x2 = random.randint(
            max(x1 + 1, width // 2),
            width
        )

        y2 = random.randint(
            max(y1 + 1, height // 2),
            height
        )

        box_area = (x2 - x1) * (y2 - y1)
        image_area = width * height

        detections.append({
            "material": material,
            "confidence": round(
                random.uniform(0.60, 0.95), 3
            ),
            "box": [x1, y1, x2, y2],
            "area_pct": round(
                box_area / image_area * 100, 2
            )
        })

    return detections


# ==========================================
# REAL YOLO DETECTION
# ==========================================

def run_yolo(image, width, height):

    model = get_model()

    results = model.predict(
        source=image,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            class_name = str(
                model.names[class_id]
            ).lower().strip()

            material = CLASS_MAP.get(class_name)

            if material is None:
                continue

            coords = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = [
                int(round(float(v))) for v in coords
            ]

            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))
            y2 = max(0, min(height, y2))

            box_area = (
                max(0, x2 - x1) *
                max(0, y2 - y1)
            )

            area_pct = (
                box_area / (width * height)
            ) * 100

            detections.append({
                "material": material,
                "confidence": round(confidence, 3),
                "box": [x1, y1, x2, y2],
                "area_pct": round(area_pct, 2)
            })

    return detections


# ==========================================
# CONDITION INPUT VALIDATION
# ==========================================

def validate_material_conditions(material_conditions):

    if material_conditions is None:
        return {}

    try:
        conditions = json.loads(material_conditions)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="material_conditions must contain valid JSON."
        )

    if not isinstance(conditions, dict):
        raise HTTPException(
            status_code=400,
            detail="material_conditions must be a JSON object."
        )

    for material, values in conditions.items():

        if material not in VALID_MATERIALS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported material: {material}"
            )

        if not isinstance(values, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Conditions for {material} must be an object."
            )

        condition = values.get("condition")
        contamination = values.get("contamination")

        if condition not in [
            "intact",
            "damaged",
            "mixed"
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid condition for {material}."
            )

        if contamination not in [
            "low",
            "high"
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid contamination for {material}."
            )

    return conditions


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
def home():

    return {
        "project": "RE:BUILD",
        "service": "ML Material Detection, Analysis and Recovery API",
        "status": "running",
        "version": "1.3.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "mock_mode": MOCK_MODE,
        "model_path": MODEL_PATH
    }


# ==========================================
# COMPLETE ML PIPELINE
# ==========================================

@app.post("/detect")
async def detect_materials(

    file: UploadFile = File(...),

    material_conditions: str = Form(default="{}")

):

    image, width, height = await read_image(file)

    conditions = validate_material_conditions(
        material_conditions
    )

    try:

        # STEP 1: YOLO MATERIAL DETECTION

        if MOCK_MODE:

            detections = mock_detection(
                width,
                height
            )

        else:

            detections = run_yolo(
                image,
                width,
                height
            )

        # STEP 2: MATERIAL ANALYSIS
        # STEP 3: RECOVERY PREDICTION

        material_analysis = analyze_materials(
            detections,
            conditions
        )

        # STEP 4: FINAL RESPONSE

        return {
            "success": True,
            "project": "RE:BUILD",
            "image_width": width,
            "image_height": height,
            "detections": detections,
            "material_analysis": material_analysis,
            "analysis_status": "completed"
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(error)}"
        )



