def apply_confidence_gate(label, confidence, threshold):
    
    label = label.lower()

    # Adaptive threshold (slightly increased for real data)
    dynamic_threshold = threshold + 0.02

    if confidence >= dynamic_threshold and "stop" in label:
        status = "VALID"
    else:
        status = "IGNORE"

    return {
        "label": label,
        "confidence": confidence,
        "threshold": dynamic_threshold,
        "status": status
    }