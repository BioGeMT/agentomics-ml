"""
Pickle-based serialization for step snapshots.
Simple, robust, and handles any Python object.
"""
import pickle
from typing import Any


def serialize_object(obj: Any) -> bytes:
    """
    Serialize any Python object to bytes using pickle.
    
    Args:
        obj: Any Python object to serialize
        
    Returns:
        Pickled bytes representation
    """
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_object(data: bytes) -> Any:
    """
    Deserialize bytes back to Python object using pickle.
    
    Args:
        data: Pickled bytes from serialize_object
        
    Returns:
        Reconstructed Python object
    """
    return pickle.loads(data)

