from __future__ import annotations

import time
from typing import Callable, TypeVar


T = TypeVar("T")


def retry_call(
    function: Callable[[], T],
    retries: int,
    base_delay: float = 1.0,
) -> T:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return function()
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError(
        f"Operation failed after {retries + 1} attempts"
    ) from last_error