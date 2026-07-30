"""Retrieval-event log: the flywheel input for learned retrieval.

Every context-pack build appends one JSONL record of what was asked and what
was returned. This is the dataset a learned ranking stage trains on — the
role ditto's ``retrieval_events`` table plays for its weight-predictor MLP —
in vouch's shape: a plaintext file beside the KB, local-only, never a cloud
row. Positive labels arrive later from the lifecycle (a ``kb_confirm``
re-citation of a claim that an event surfaced is a human-verified relevance
judgment, a label source ditto does not have).

Properties, all deliberate:

* **local-only telemetry** — the file is gitignored (the init template lists
  it; first write appends the ignore line for pre-existing KBs) and never
  committed;
* **masked** — queries pass through ``mask_secrets`` before touching disk,
  same rule as the capture buffer;
* **bounded** — a size cap rotates the file to ``.1`` (one generation kept),
  so the log never grows without limit;
* **never load-bearing** — logging failure is swallowed; retrieval results
  are identical with the log on, off, or broken. ``retrieval.events.enabled:
  false`` turns it off.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import yaml

from .config_coerce import coerce_bool
from .secrets import mask_secrets
from .storage import KBStore

logger = logging.getLogger(__name__)

FILENAME = "retrieval_events.jsonl"
ROTATED_FILENAME = "retrieval_events.jsonl.1"
DEFAULT_ENABLED = True
DEFAULT_MAX_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class EventsConfig:
    enabled: bool = DEFAULT_ENABLED
    max_bytes: int = DEFAULT_MAX_BYTES


def load_events_config(store: KBStore) -> EventsConfig:
    """Read ``retrieval.events`` from config.yaml; fall back to defaults."""
    try:
        loaded = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return EventsConfig()
    if not isinstance(loaded, dict):
        return EventsConfig()
    retrieval = loaded.get("retrieval")
    raw = retrieval.get("events") if isinstance(retrieval, dict) else None
    if not isinstance(raw, dict):
        return EventsConfig()
    try:
        max_bytes = int(raw.get("max_bytes", DEFAULT_MAX_BYTES))
    except (TypeError, ValueError):
        max_bytes = DEFAULT_MAX_BYTES
    return EventsConfig(
        enabled=coerce_bool(raw.get("enabled", DEFAULT_ENABLED), DEFAULT_ENABLED),
        max_bytes=max_bytes,
    )


def _events_path(store: KBStore) -> Any:
    return store.kb_dir / FILENAME


def _ensure_ignored(store: KBStore) -> None:
    """Append the log to .vouch/.gitignore when a pre-existing KB lacks it.

    New KBs get the line from the init template; this backfills the rest so
    telemetry can never end up committed. Best-effort: an unwritable
    .gitignore must not break retrieval.
    """
    gi = store.kb_dir / ".gitignore"
    try:
        text = gi.read_text(encoding="utf-8") if gi.exists() else ""
        if FILENAME in text:
            return
        if text and not text.endswith("\n"):
            text += "\n"
        gi.write_text(
            text + FILENAME + "\n" + ROTATED_FILENAME + "\n", encoding="utf-8"
        )
    except OSError:
        return


def log_event(
    store: KBStore,
    *,
    query: str,
    backend: str,
    limit: int,
    budget_chars: int | None,
    items: list[dict[str, Any]],
    config: EventsConfig | None = None,
) -> bool:
    """Append one retrieval event. Returns True when a record was written.

    ``items`` is the returned hit list; only (type, id, score) are kept —
    summaries would bloat the log and re-copy KB content into telemetry.
    """
    cfg = config or load_events_config(store)
    if not cfg.enabled:
        return False
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "query": mask_secrets(query),
        "backend": backend,
        "limit": limit,
        "budget_chars": budget_chars,
        "items": [
            {
                "type": str(it.get("type", "")),
                "id": str(it.get("id", "")),
                "score": it.get("score", 0.0),
            }
            for it in items
        ],
    }
    path = _events_path(store)
    try:
        _ensure_ignored(store)
        if path.exists() and path.stat().st_size >= cfg.max_bytes:
            path.replace(store.kb_dir / ROTATED_FILENAME)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.debug("retrieval_events: write failed (%s)", e)
        return False
    return True


def read_events(store: KBStore, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Parsed events, oldest first; ``limit`` keeps the newest N."""
    path = _events_path(store)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    except OSError:
        return []
    if limit is None:
        return out
    # `out[-0:]` is `out[:]`, so a zero limit would return the whole log.
    return out[-limit:] if limit > 0 else []
