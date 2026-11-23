from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_progress_event(
    path: str | Path,
    *,
    elapsed_seconds: float,
    stage: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be finite and non-negative")
    if not stage or not status:
        raise ValueError("stage and status must be non-empty")

    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "stage": stage,
        "status": status,
        "details": details or {},
    }
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
    return event


def record_progress_event(
    log_path: str | Path,
    *,
    elapsed_seconds: float,
    stage: str,
    progress_status: str,
    **details: Any,
) -> dict[str, Any]:
    return append_progress_event(
        log_path,
        elapsed_seconds=elapsed_seconds,
        stage=stage,
        status=progress_status,
        details=details,
    )
