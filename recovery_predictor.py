
# ==========================================
# RE:BUILD - MATERIAL RECOVERY PREDICTOR
# ==========================================

MATERIAL_RULES = {
    "concrete": {
        "intact": {
            "recommendation": "Reuse",
            "method": "Direct structural reuse after engineering inspection",
            "reason": "Intact concrete elements may be suitable for reuse."
        },
        "damaged": {
            "recommendation": "Recycle",
            "method": "Crush into recycled aggregates",
            "reason": "Damaged concrete can be processed into recycled aggregate."
        },
        "mixed": {
            "recommendation": "Recycle after sorting",
            "method": "Separate and process concrete waste",
            "reason": "Mixed concrete waste requires sorting before processing."
        }
    },

    "brick": {
        "intact": {
            "recommendation": "Reuse",
            "method": "Clean and recover individual bricks",
            "reason": "Intact bricks may be reused in construction."
        },
        "damaged": {
            "recommendation": "Recycle",
            "method": "Crush into recycled aggregate",
            "reason": "Broken bricks can be crushed for secondary applications."
        },
        "mixed": {
            "recommendation": "Recycle after sorting",
            "method": "Separate bricks from mixed debris",
            "reason": "Sorting improves brick recovery potential."
        }
    },

    "steel": {
        "intact": {
            "recommendation": "Reuse",
            "method": "Recover steel components after inspection",
            "reason": "Suitable steel components may be reused."
        },
        "damaged": {
            "recommendation": "Recycle",
            "method": "Recover and send for scrap metal recycling",
            "reason": "Steel scrap can be melted and recycled."
        },
        "mixed": {
            "recommendation": "Recycle after sorting",
            "method": "Separate steel from other waste",
            "reason": "Steel should be segregated before recycling."
        }
    },

    "wood": {
        "intact": {
            "recommendation": "Reuse",
            "method": "Recover and reuse wooden components",
            "reason": "Intact wood may be reused after inspection."
        },
        "damaged": {
            "recommendation": "Recycle",
            "method": "Process suitable wood waste into wood products",
            "reason": "Suitable damaged wood can be processed for recycling."
        },
        "mixed": {
            "recommendation": "Recycle after sorting",
            "method": "Separate treated and untreated wood",
            "reason": "Wood must be sorted before selecting a recovery route."
        }
    },

    "glass": {
        "intact": {
            "recommendation": "Reuse",
            "method": "Recover intact glass panels",
            "reason": "Undamaged glass panels may be reused."
        },
        "damaged": {
            "recommendation": "Recycle",
            "method": "Collect and process suitable glass",
            "reason": "Suitable glass fragments can be recycled."
        },
        "mixed": {
            "recommendation": "Recycle after sorting",
            "method": "Separate glass from mixed construction waste",
            "reason": "Glass requires appropriate sorting before recycling."
        }
    },

    "tiles": {
        "intact": {
            "recommendation": "Reuse",
            "method": "Recover and reinstall intact tiles",
            "reason": "Undamaged tiles may be reused."
        },
        "damaged": {
            "recommendation": "Recycle",
            "method": "Process suitable tile waste into secondary aggregates",
            "reason": "Broken tiles may be used in suitable secondary applications."
        },
        "mixed": {
            "recommendation": "Recycle after sorting",
            "method": "Separate tiles from mixed debris",
            "reason": "Sorting helps identify recoverable tile waste."
        }
    }
}


def predict_recovery(material, condition, contamination="low"):

    material = material.lower().strip()
    condition = condition.lower().strip()
    contamination = contamination.lower().strip()

    if material not in MATERIAL_RULES:
        raise ValueError("Unsupported material type.")

    if condition not in ["intact", "damaged", "mixed"]:
        raise ValueError(
            "Condition must be intact, damaged or mixed."
        )

    if contamination not in ["low", "high"]:
        raise ValueError(
            "Contamination must be low or high."
        )

    result = MATERIAL_RULES[material][condition].copy()

    if contamination == "high":
        result["recommendation"] = "Further inspection required"
        result["method"] = "Sort, clean and inspect before recovery"
        result["reason"] = (
            "High contamination may affect reuse and recycling "
            "suitability."
        )

    return {
        "material": material,
        "condition": condition,
        "contamination": contamination,
        "recovery_prediction": result,
        "assessment_type": "Rule-based preliminary assessment",
        "requires_expert_inspection": True
    }
