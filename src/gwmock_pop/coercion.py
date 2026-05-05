"""Helpers for coercing population records into NumPy-backed values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def coerce_to_numpy(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert array-backed record values to NumPy arrays or Python scalars.

    This helper is intended for consumer boundaries where ``gwmock-pop`` output
    should no longer require a JAX runtime. Array-valued inputs are materialized
    as ``numpy.ndarray`` values, while 0-D array scalars are unwrapped to native
    Python scalars via :meth:`numpy.ndarray.item`.

    Args:
        record: Mapping of parameter names to array-like or scalar values.

    Returns:
        A new mapping where JAX and NumPy values are converted to NumPy-backed
        objects. Non-array values are passed through unchanged.
    """
    coerced: dict[str, Any] = {}
    for name, value in record.items():
        if not hasattr(value, "__array__") and not isinstance(value, (np.generic, np.ndarray)):
            coerced[name] = value
            continue

        numpy_value = np.asarray(value)
        coerced[name] = numpy_value.item() if numpy_value.ndim == 0 else numpy_value
    return coerced
