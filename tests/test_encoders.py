import numpy as np
from src.utils.encoders import custom_jsonable_encoder

def test_encode_numpy_int64():
    """
    Test encoding of numpy.int64 values.
    """
    obj = np.int64(42)
    encoded = custom_jsonable_encoder(obj)
    assert encoded == 42
    assert isinstance(encoded, int)

def test_encode_numpy_float64():
    """
    Test encoding of numpy.float64 values.
    """
    obj = np.float64(42.42)
    encoded = custom_jsonable_encoder(obj)
    assert encoded == 42.42
    assert isinstance(encoded, float)

def test_encode_numpy_array():
    """
    Test encoding of numpy.ndarray values.
    """
    obj = np.array([1, 2, 3])
    encoded = custom_jsonable_encoder(obj)
    assert encoded == [1, 2, 3]
    assert isinstance(encoded, list)

def test_encode_nested_structure():
    """
    Test encoding of a complex nested structure containing numpy types.
    """
    obj = {
        "int_value": np.int64(42),
        "float_value": np.float64(42.42),
        "array_value": np.array([1, 2, 3]),
    }
    encoded = custom_jsonable_encoder(obj)
    assert encoded == {
        "int_value": 42,
        "float_value": 42.42,
        "array_value": [1, 2, 3],
    }
