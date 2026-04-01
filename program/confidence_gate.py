# Confidence Gate + Adaptive Threshold Logic

VALID_KEYWORD = "stop"

def calculate_dynamic_threshold(confidence, base_threshold):
    """
    Adaptive Threshold:
    Adjust threshold based on confidence strength
    """

    # High confidence → stricter decision
    if confidence >= 0.90:
        return base_threshold + 0.03

    # Medium confidence → moderate adjustment
    elif confidence >= 0.80:
        return base_threshold + 0.02

    # Low confidence → keep threshold same
    else:
        return base_threshold


def apply_confidence_gate(label, confidence, threshold):
    """
    Final decision layer:
    Uses adaptive threshold + keyword validation
    """

    label = label.lower()

    # 🔥 Apply adaptive threshold
    dynamic_threshold = calculate_dynamic_threshold(confidence, threshold)

    # Decision logic
    if confidence >= dynamic_threshold and VALID_KEYWORD in label:
        status = "VALID"
    else:
        status = "IGNORE"

    return {
        "label": label,
        "confidence": confidence,
        "threshold": dynamic_threshold,
        "status": status
    }