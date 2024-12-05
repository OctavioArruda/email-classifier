from fastapi.encoders import jsonable_encoder
import numpy as np

def custom_jsonable_encoder(obj):
    """
    Custom JSON encoder that handles numpy types and nested structures.
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Recursively handle dictionary values
        return {key: custom_jsonable_encoder(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Recursively handle lists
        return [custom_jsonable_encoder(item) for item in obj]
    return jsonable_encoder(obj)
