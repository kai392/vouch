"""Read-only introspection over the retrieval ranking pipeline — issue #432.

``_retrieve`` returns each hit as ``(kind, id, summary, score, backend)``: one
opaque score and a backend label. A reviewer tuning fusion, the reranker, or
the recency / pages-first signals has no way to see *why* an artifact surfaced
or got dropped — how much came from lexical vs. semantic rank, what the RRF
contribution was, whether a rescoring stage moved it, or which gate cut it.

This module re-runs those stages against the same helpers ``context`` uses,
snapshotting every candidate's rank and score after each one. A candidate
present in one snapshot and absent from the next was removed by that stage,
which is what the reported gate names.

The composition is deliberate rather than a copy of any single caller: it
chains ``_retrieve``'s ranking stages, the lifecycle gate ``search_kb``
applies, and the budget / citation gates ``build_context_pack`` applies, so
one call explains every stage an artifact can die at. Scope filtering runs
without a limit and truncation is a separate ``limit`` stage — the same
"scope first so status filtering can refill the window" ordering ``search_kb``
uses, and it keeps a candidate lost to truncation attributable instead of
folding it into the scope filter.

Two stage shapes matter when reading a breakdown:

* ``recency`` and ``pages_first`` are rescoring-only — the candidate set is
  unchanged, so their signal is the score delta.
* ``rerank`` and ``strategy`` are ordering-only — scores are untouched, so
  their signal is the rank delta.

``strategy`` is the pluggable final reorder; it is reported as a stage so a
shipped ranking plugin's effect is visible rather than folded into the score
it did not change.

Read-only by construction: every helper called here is one the read path
already uses, and nothing writes, proposes, or mutates the KB. Viewer scoping
runs through the same ``filter_hits`` as ``kb.context``: a candidate the
viewer cannot retrieve is still listed, because naming the gate that hid it is
the point of the report, but it is listed without its summary — the content
``kb.search`` and ``kb.context`` withhold for that viewer is withheld here too.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from . import index_db
from .context import (
    _configured_backend,
    _configured_pages_first,
    _configured_recency,
    _configured_rerank,
    _configured_strategy,
    _filter_live_hits,
    _maybe_pages_first,
    _maybe_recency,
    _maybe_rerank,
    _maybe_strategy,
)
from .embeddings.fusion import rrf_fuse
from .scoping import ViewerContext, filter_hits, scoped_fetch_limit, viewer_from
from .storage import KBStore

Hit = tuple[str, str, str, float]
Key = tuple[str, str]

DEFAULT_LIMIT = 10

# Stage name -> the gate reported for a candidate that stage removed.
_GATE_FOR_STAGE = {
    "scope_filter": "scope-filtered",
    "status_filter": "status-filtered",
    "limit": "limit-dropped",
    "budget": "budget-dropped",
}


def _key(hit: Hit) -> Key:
    return (hit[0], hit[1])


@dataclass
class _Snapshot:
    """The candidate set after one pipeline stage."""

    stage: str
    applied: bool
    ranks: dict[Key, int] = field(default_factory=dict)
    scores: dict[Key, float] = field(default_factory=dict)

    @classmethod
    def of(cls, stage: str, hits: list[Hit], *, applied: bool = True) -> _Snapshot:
        return cls(
            stage=stage,
            applied=applied,
            ranks={_key(h): i for i, h in enumerate(hits, start=1)},
            scores={_key(h): h[3] for h in hits},
        )


def _retrieve_traced(
    store: KBStore,
    query: str,
    limit: int,
    viewer: ViewerContext,
) -> tuple[list[Hit], str, dict[Key, int], dict[Key, int], list[_Snapshot]]:
    """Re-run ``_retrieve``'s backend selection, keeping the per-retriever ranks.

    Returns the fused hits, the backend that served them, the lexical and
    semantic rank maps, and the fusion snapshot. Mirrors ``_retrieve``'s
    branching so the explanation describes the query that would actually run.
    """
    backend = _configured_backend(store)
    fetch_limit = scoped_fetch_limit(limit, viewer)
    sem: list[Hit] = []
    lex: list[Hit] = []

    def _lexical() -> list[Hit]:
        try:
            return index_db.search(store.kb_dir, query, limit=fetch_limit)
        except sqlite3.Error:
            return []

    if backend in ("auto", "hybrid"):
        sem = index_db.search_semantic(store.kb_dir, query, limit=fetch_limit)
        lex = _lexical()
        fused = rrf_fuse(sem, lex, limit=fetch_limit)
        if fused:
            used = "hybrid"
        else:
            # Both retrievers came back empty: _retrieve falls through to the
            # substring scan, which applies none of the rescoring stages.
            fused = store.search_substring(query, limit=fetch_limit)
            used = "substring"
    elif backend == "embedding":
        sem = index_db.search_semantic(store.kb_dir, query, limit=fetch_limit)
        fused, used = sem, "embedding"
    elif backend == "fts5":
        lex = _lexical()
        fused, used = lex, "fts5"
    else:
        fused, used = store.search_substring(query, limit=fetch_limit), "substring"

    lex_ranks = {_key(h): i for i, h in enumerate(lex, start=1)}
    sem_ranks = {_key(h): i for i, h in enumerate(sem, start=1)}
    return fused, used, lex_ranks, sem_ranks, [_Snapshot.of(used, fused)]


def _budget_survivors(hits: list[Hit], max_chars: int) -> list[Hit]:
    """The hits ``build_context_pack`` would keep under a ``max_chars`` budget.

    Mirrors the pack's omission pass — drop from the tail until the summary
    total fits. The pack's clipping pass shortens a summary rather than
    dropping the item, so it never changes the candidate set and is reported
    as a stage that kept everything.
    """
    kept = list(hits)
    while kept and sum(len(h[2]) for h in kept) > max_chars:
        kept.pop()
    return kept


def _is_uncited(store: KBStore, key: Key) -> bool:
    """True for a surviving claim that carries no citations.

    ``require_citations`` does not drop an item — ``build_context_pack`` keeps
    it and fails the pack (``failed: ["require_citations"]``). So this is not a
    drop stage: it renames the gate on the candidate that is *responsible* for
    that failure, which is the artifact a reviewer needs to find.

    No missing-artifact guard: only candidates that survived ``status_filter``
    reach here, and that stage already dropped every claim it could not read.

    Defensive today — ``Claim`` rejects ``evidence=[]`` on the model, which
    closes every write path (``models.py``: "claim must cite at least one
    Source or Evidence id"), so no stored claim can be uncited. It mirrors the
    check ``build_context_pack`` still makes, and starts reporting the moment
    that invariant is relaxed rather than silently reporting ``kept``.
    """
    if key[0] != "claim":
        return False
    return not store.get_claim(key[1]).evidence


def _stage_rows(
    key: Key,
    snapshots: list[_Snapshot],
) -> tuple[list[dict[str, Any]], str]:
    """Per-stage rows for one candidate, plus the gate that decided its fate."""
    rows: list[dict[str, Any]] = []
    gate = "kept"
    prev_rank: int | None = None
    prev_score: float | None = None

    for snap in snapshots:
        if key not in snap.ranks:
            # Absent here but present in the previous snapshot -> this stage
            # removed it. A stage that did not run cannot have dropped it.
            if prev_rank is not None and snap.applied:
                gate = _GATE_FOR_STAGE.get(snap.stage, f"{snap.stage}-dropped")
            break
        rank, score = snap.ranks[key], snap.scores[key]
        rows.append({
            "stage": snap.stage,
            "applied": snap.applied,
            "rank": rank,
            "score": round(score, 6),
            "rank_delta": None if prev_rank is None else prev_rank - rank,
            "score_delta": None if prev_score is None else round(score - prev_score, 6),
        })
        prev_rank, prev_score = rank, score

    return rows, gate


def explain_ranking(
    store: KBStore,
    *,
    query: str,
    limit: int = DEFAULT_LIMIT,
    max_chars: int | None = None,
    require_citations: bool = False,
    project: str | None = None,
    agent: str | None = None,
) -> dict[str, Any]:
    """Explain why each candidate for *query* ranked where it did.

    Returns ``{"query", "limit", "viewer", "retrieval", "candidates"}``.
    Each candidate carries its lexical and semantic rank, its RRF
    contribution, a row per pipeline stage, and the gate that kept or
    dropped it. Read-only — no write path is touched.
    """
    if limit < 0:
        raise ValueError("limit must be >= 0")

    viewer = viewer_from(config_path=store.config_path, project=project, agent=agent)
    hits, used, lex_ranks, sem_ranks, snapshots = _retrieve_traced(
        store, query, limit, viewer
    )
    rrf_scores = {_key(h): h[3] for h in hits}

    scoped = filter_hits(store, hits, viewer)
    snapshots.append(_Snapshot.of("scope_filter", scoped))

    live = _filter_live_hits(store, scoped)
    snapshots.append(_Snapshot.of("status_filter", live))

    recency_on, _half_life = _configured_recency(store)
    rescored = _maybe_recency(store, hits=live)
    snapshots.append(_Snapshot.of("recency", rescored, applied=recency_on))

    pages_first_on, _boost = _configured_pages_first(store)
    boosted = _maybe_pages_first(store, hits=rescored)
    snapshots.append(_Snapshot.of("pages_first", boosted, applied=pages_first_on))

    rerank_on, rerank_top_k = _configured_rerank(store, limit=limit)
    reranked = _maybe_rerank(store, query=query, hits=boosted, limit=limit)
    snapshots.append(_Snapshot.of("rerank", reranked, applied=rerank_on))

    # _maybe_strategy speaks the 5-tuple retrieval shape (backend appended);
    # carry `used` through and drop it again so the stage list stays uniform.
    strategy_name = _configured_strategy(store)
    ordered = [
        (k, i, s, sc)
        for k, i, s, sc, _be in _maybe_strategy(
            store,
            query=query,
            hits=[(k, i, s, sc, used) for k, i, s, sc in reranked],
            limit=limit,
        )
    ]
    snapshots.append(
        _Snapshot.of("strategy", ordered, applied=strategy_name is not None)
    )

    windowed = ordered[:limit]
    snapshots.append(_Snapshot.of("limit", windowed))

    if max_chars is not None:
        windowed = _budget_survivors(windowed, max_chars)
        snapshots.append(_Snapshot.of("budget", windowed))

    # Every candidate the pipeline ever saw, in the order fusion produced them,
    # so a dropped artifact is still explained rather than silently missing.
    # Summaries come from the *scoped* set: a candidate the viewer cannot
    # retrieve keeps its gate attribution — that is the point of the report —
    # but not its text. `kb.search` and `kb.context` withhold that summary for
    # this viewer, so this surface must not hand it back.
    candidates: list[dict[str, Any]] = []
    summaries = {_key(h): h[2] for h in scoped}
    for hit in hits:
        key = _key(hit)
        rows, gate = _stage_rows(key, snapshots)
        if gate == "kept" and require_citations and _is_uncited(store, key):
            gate = "uncited"
        candidates.append({
            "kind": key[0],
            "id": key[1],
            "summary": summaries.get(key, ""),
            "lexical_rank": lex_ranks.get(key),
            "semantic_rank": sem_ranks.get(key),
            "rrf_contribution": round(rrf_scores.get(key, 0.0), 6),
            "stages": rows,
            "gate": gate,
        })

    return {
        "query": query,
        "limit": limit,
        "viewer": {"project": viewer.project, "agent": viewer.agent},
        "retrieval": {
            "configured": _configured_backend(store),
            "used": used,
            "semantic_available": index_db.semantic_search_available(),
            "stages": {
                "fusion": used == "hybrid",
                "recency": recency_on,
                "pages_first": pages_first_on,
                "rerank": rerank_on,
                "rerank_top_k": rerank_top_k if rerank_on else None,
                "strategy": strategy_name,
                "budget": max_chars,
                "require_citations": require_citations,
            },
        },
        "candidates": candidates,
    }
