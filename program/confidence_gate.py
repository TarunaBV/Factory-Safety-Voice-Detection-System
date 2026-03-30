# This function is responsible for making the final decision
# after the model predicts a label and confidence score

def apply_confidence_gate(label, confidence, threshold):
    
    # Convert label to lowercase for consistency
    label = label.lower()

    # Condition 1: confidence should be greater than threshold
    # Condition 2: predicted label should contain the keyword "stop"
    # (since our dataset is based on STOP vs NOISE)
    if confidence >= threshold and "stop" in label:
        status = "VALID"   # Accept as true detection
    else:
        status = "IGNORE"  # Reject as noise or low confidence

    # Return structured output
    return {
        "label": label,
        "confidence": confidence,
        "threshold": threshold,
        "status": status
    }