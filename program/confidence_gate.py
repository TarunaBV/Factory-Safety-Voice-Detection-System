# Confidence Gate + Adaptive Threshold Logic

VALID_KEYWORDS = ["stop", "help", "fire"]


def calculate_dynamic_threshold(confidence, base_threshold):
    """
    Adaptive Threshold:
    Adjust threshold based on confidence strength
    """

    if confidence >= 0.90:
        return base_threshold + 0.03
    elif confidence >= 0.80:
        return base_threshold + 0.02
    else:
        return base_threshold


def apply_confidence_gate(label, confidence, threshold):
    """
    Final decision layer:
    Uses adaptive threshold + keyword validation
    """

    label = label.lower().strip()

    dynamic_threshold = calculate_dynamic_threshold(confidence, threshold)

    # safer keyword check
    if confidence >= dynamic_threshold and label in VALID_KEYWORDS:
        status = "VALID"
    else:
        status = "IGNORE"

    return {
        "label": label,
        "confidence": confidence,
        "threshold": dynamic_threshold,
        "status": status
    }