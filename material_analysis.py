
from collections import defaultdict

from recovery_predictor import predict_recovery


# ==========================================
# RE:BUILD - MATERIAL ANALYSIS MODULE
# ==========================================

VALID_MATERIALS = [
    "concrete",
    "brick",
    "steel",
    "wood",
    "glass",
    "tiles"
]


# Default condition settings
# These are temporary defaults until the frontend
# collects actual user inputs.

DEFAULT_CONDITION = "mixed"
DEFAULT_CONTAMINATION = "low"


# ==========================================
# MATERIAL ANALYSIS FUNCTION
# ==========================================

def analyze_materials(detections, material_conditions=None):
    """
    Converts YOLO detections into material-wise analysis
    and generates recovery predictions.

    detections:
        List of YOLO detection dictionaries.

    material_conditions:
        Optional dictionary containing condition and
        contamination for each material.

    Example:
        {
            "brick": {
                "condition": "intact",
                "contamination": "low"
            }
        }
    """

    if material_conditions is None:
        material_conditions = {}

    if not isinstance(material_conditions, dict):
        raise ValueError(
            "material_conditions must be a dictionary."
        )

    material_summary = defaultdict(lambda: {
        "count": 0,
        "confidence_sum": 0.0,
        "visible_area_pct": 0.0
    })

    # ==========================================
    # STEP 1: GROUP YOLO DETECTIONS
    # ==========================================

    for detection in detections:

        material = str(
            detection.get("material", "")
        ).lower().strip()

        if material not in VALID_MATERIALS:
            continue

        confidence = float(
            detection.get("confidence", 0)
        )

        area_pct = float(
            detection.get("area_pct", 0)
        )

        if not (0 <= confidence <= 1):
            continue

        if area_pct < 0:
            continue

        summary = material_summary[material]

        summary["count"] += 1
        summary["confidence_sum"] += confidence
        summary["visible_area_pct"] += area_pct

    # ==========================================
    # STEP 2: MATERIAL-WISE ANALYSIS
    # ==========================================

    results = {}

    for material, data in material_summary.items():

        count = data["count"]

        average_confidence = round(
            data["confidence_sum"] / count,
            3
        )

        visible_area = round(
            data["visible_area_pct"],
            2
        )

        # ==========================================
        # STEP 3: GET USER CONDITION INPUT
        # ==========================================

        condition_data = material_conditions.get(
            material, {}
        )

        condition = condition_data.get(
            "condition",
            DEFAULT_CONDITION
        )

        contamination = condition_data.get(
            "contamination",
            DEFAULT_CONTAMINATION
        )

        # ==========================================
        # STEP 4: RECOVERY PREDICTION
        # ==========================================

        recovery_result = predict_recovery(
            material=material,
            condition=condition,
            contamination=contamination
        )

        # ==========================================
        # STEP 5: COMBINE RESULTS
        # ==========================================

        results[material] = {
            "count": count,

            "average_confidence": average_confidence,

            "visible_area_pct_sum": visible_area,

            "condition": condition,

            "contamination": contamination,

            "recovery": recovery_result[
                "recovery_prediction"
            ],

            "assessment_type": recovery_result[
                "assessment_type"
            ],

            "requires_expert_inspection": recovery_result[
                "requires_expert_inspection"
            ]
        }

    # ==========================================
    # FINAL RESPONSE
    # ==========================================

    return {
        "total_detections": sum(
            item["count"] for item in results.values()
        ),

        "materials_detected": len(results),

        "material_summary": results
    }
