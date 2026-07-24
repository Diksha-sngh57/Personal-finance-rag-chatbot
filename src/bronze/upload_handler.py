"""
Persists user-uploaded statement files (from the Streamlit UI) to
data/uploads/ with collision-safe naming, before Bronze ever sees them.

Bronze's own idempotency (content-hash manifest, src/bronze/pipeline.py)
already guards against re-ingesting the same BYTES twice. This module
guards against a different problem one layer up: two DIFFERENT files
that happen to share the same original filename (e.g. re-exporting
"statement.pdf" from a banking app twice, a week apart) would otherwise
silently overwrite one another on disk before Bronze gets a chance to
see the first one.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from src.config import UPLOADS_DIR, ensure_dir

logger = logging.getLogger(__name__)


def save_uploaded_file(uploaded_file) -> Path:
    """
    Persists a Streamlit UploadedFile to UPLOADS_DIR.

    Naming: <original_stem>__<8-char-uuid><original_suffix>. The uuid
    suffix is ALWAYS appended, not only when a collision is detected —
    checking "does this filename already exist" first and renaming only
    on collision has a TOCTOU race if this is ever called concurrently
    (e.g. two browser tabs uploading at once). Always-unique naming has
    no such race, at the minor cost of slightly less readable filenames
    on disk.
    """
    ensure_dir(UPLOADS_DIR)

    original_name = uploaded_file.name
    suffix = Path(original_name).suffix
    stem = Path(original_name).stem

    unique_name = f"{stem}__{uuid.uuid4().hex[:8]}{suffix}"
    destination = UPLOADS_DIR / unique_name

    raw_bytes = uploaded_file.getvalue()
    with open(destination, "wb") as f:
        f.write(raw_bytes)

    logger.info(
        "Persisted upload '%s' -> '%s' (%d bytes)",
        original_name, destination.name, len(raw_bytes),
    )
    return destination