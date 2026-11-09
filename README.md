
# RE:BUILD ML Service

AI-powered construction material detection using
FastAPI, YOLOv8 and OpenCV.

## Supported materials

- Concrete
- Brick
- Steel
- Wood
- Glass
- Tiles

## Installation

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

# RE:BUILD — AI-Powered Construction Material Detection

An AI-powered construction and demolition waste analysis system that identifies building materials using computer vision and provides preliminary material recovery recommendations.

**Role:** Machine Learning Lead  
**Project:** RE:BUILD  
**Technologies:** Python, YOLOv8, FastAPI, OpenCV

## Project Overview

RE:BUILD aims to improve construction and demolition waste management through AI-powered material identification and recovery assessment.

The ML module processes uploaded images, detects construction materials, analyzes detected materials, and generates preliminary recovery recommendations.

## Key Features

- **Material Detection:** YOLOv8-based object detection.
- **Six Material Classes:** Concrete, brick, steel, wood, glass, and tiles.
- **Material Analysis:** Groups detections by material and calculates detection counts, average confidence, and bounding-box area.
- **Recovery Recommendations:** Rule-based recommendations based on material condition and contamination.
- **REST API:** FastAPI endpoint for image-based material detection and analysis.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | ML development |
| YOLOv8 | Object detection |
| OpenCV | Image processing |
| FastAPI | Prediction API |
| Ultralytics | YOLO model training and inference |

## System Architecture

```text
Construction Site Image
          |
          v
     FastAPI Endpoint
          |
          v
     Image Processing
          |
          v
     Trained YOLOv8 Model
          |
          v
    Material Detection
          |
          v
    Material Analysis
          |
          v
  Recovery Recommendation
          |
          v
       JSON Response
```

## Supported Material Classes

1. Concrete
2. Brick
3. Steel
4. Wood
5. Glass
6. Tiles

## Recovery Assessment

The recovery predictor uses material-specific rules and two input attributes:

**Condition**
- `intact`
- `damaged`
- `mixed`

**Contamination**
- `low`
- `high`

The system generates preliminary recommendations such as reuse, recycling, or further inspection.

All recommendations require expert inspection before practical implementation.

## API Documentation

### POST `/detect`

Accepts an image and material-condition information through multipart form data.

**Request fields**

| Field | Type | Description |
|---|---|---|
| `file` | Image | JPEG, PNG, or WEBP image |
| `material_conditions` | JSON string | Material-wise condition and contamination |

**Example material conditions**

```json
{
  "brick": {
    "condition": "intact",
    "contamination": "low"
  },
  "steel": {
    "condition": "damaged",
    "contamination": "low"
  }
}
```

**Response structure**

The API returns:
- Success status
- Image dimensions
- Material detections
- Confidence scores
- Bounding-box coordinates
- Material-wise analysis
- Preliminary recovery recommendations

### Interactive API Documentation

Once the service is running, access:

`http://127.0.0.1:8000/docs`

## Installation

Clone the repository:

```bash
git clone https://github.com/ishanigholap-sketch/REBUILD-ML-Material-Detection.git
cd REBUILD-ML-Material-Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Training

Prepare your dataset and configure its YAML file.

Run:

```bash
python train.py --data path/to/data.yaml
```

Training runs for 30 epochs.

Best weights are saved in:

```text
runs/detect/rebuild-materials/weights/best.pt
```

The trained weights are not included in this repository. Use your own appropriately licensed trained weights and configure `MODEL_PATH`.

## Running the API

Configure your local `.env` file:

```env
MOCK_MODE=false
MODEL_PATH=best.pt
MAX_IMAGE_MB=10
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Start FastAPI:

```bash
uvicorn main:app --reload --port 8000
```

Open the Swagger documentation at:

`http://127.0.0.1:8000/docs`

## Limitations

- Recovery recommendations are preliminary and rule-based.
- Physical material quantity estimation in kilograms is not currently implemented.
- Bounding-box area is an image-space measurement, not a physical quantity estimate.
- Predictions require expert verification before real-world waste recovery decisions.

## My Contribution

As the Machine Learning Lead, I worked on:

- Dataset preparation and organization.
- YOLOv8 model training for six construction material classes.
- Development of the FastAPI prediction service.
- Image processing and material detection.
- Material-wise analysis and recovery recommendation logic.
- API testing and preparation of the ML integration contract.

## Project Context

RE:BUILD is a collaborative project focused on AI-powered construction and demolition waste recovery planning.

This repository contains the machine learning component developed for the project.