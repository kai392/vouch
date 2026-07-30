"""Read-only ranking introspection — issue #432."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from vouch import explain_ranking as er
from vouch import health
from vouch.models import (
    ArtifactScope,
    Claim,
    ClaimStatus,
    Page,
    PageStatus,
    PageType,
    Visibility,
)
from vouch.storage import KBStore


def _write_cfg(store: KBStore, **retrieval: object) -> None:
    (store.kb_dir / "config.yaml").write_text(
        yaml.safe_dump({"retrieval": retrieval}), encoding="utf-8"
    )


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KBStore:
    kb = KBStore.init(tmp_path)
    monkeypatch.chdir(kb.root)
    src = kb.put_source(b"auth notes")
    kb.put_claim(Claim(id="c1", text="auth uses jwt tokens", evidence=[src.id]))
    kb.put_claim(Claim(id="c2", text="jwt rotation is manual", evidence=[src.id]))
    health.rebuild_index(kb)
    return kb


def _by_id(result: dict) -> dict[str, dict]:
    return {c["id"]: c for c in result["candidates"]}


def _stages(candidate: dict) -> list[str]:
    return [row["stage"] for row in candidate["stages"]]


# --- the fused-only query the issue asks for -----------------------------


def test_fused_only_query_reports_per_retriever_ranks(store: KBStore) -> None:
    """A fused query exposes lexical rank, semantic rank and the rrf score."""
    result = er.explain_ranking(store, query="jwt", limit=5)

    assert result["retrieval"]["used"] == "hybrid"
    assert result["retrieval"]["stages"]["fusion"] is True
    cand = _by_id(result)["c1"]
    # fts5 served this query; the embedding retriever is absent without extras,
    # which is exactly the asymmetry the breakdown is supposed to make visible.
    assert cand["lexical_rank"] is not None
    assert cand["rrf_contribution"] > 0
    assert cand["gate"] == "kept"
    assert _stages(cand)[0] == "hybrid"
    assert "limit" in _stages(cand)


def test_first_stage_row_has_no_deltas(store: KBStore) -> None:
    """Nothing precedes fusion, so its deltas are null rather than zero."""
    first = _by_id(er.explain_ranking(store, query="jwt"))["c1"]["stages"][0]
    assert first["rank_delta"] is None
    assert first["score_delta"] is None


# --- gate-dropped candidates ---------------------------------------------


@pytest.mark.parametrize(
    "retracted", [ClaimStatus.ARCHIVED, ClaimStatus.SUPERSEDED, ClaimStatus.REDACTED]
)
def test_retracted_claim_reports_status_filtered(
    store: KBStore, retracted: ClaimStatus
) -> None:
    store.put_claim(Claim(
        id="c-dead", text="jwt replaced by sessions",
        evidence=[store.list_sources()[0].id], status=retracted,
    ))
    health.rebuild_index(store)

    cand = _by_id(er.explain_ranking(store, query="jwt", limit=5))["c-dead"]
    assert cand["gate"] == "status-filtered"
    # it is explained up to the stage that cut it, not silently missing
    assert _stages(cand) == ["hybrid", "scope_filter"]


def test_archived_page_reports_status_filtered(store: KBStore) -> None:
    store.put_page(Page(
        id="p-arch", title="jwt legacy design", body="old",
        type=PageType.CONCEPT, status=PageStatus.ARCHIVED,
    ))
    health.rebuild_index(store)

    assert _by_id(er.explain_ranking(store, query="jwt", limit=5))["p-arch"]["gate"] == (
        "status-filtered"
    )


def test_truncated_candidate_reports_limit_dropped(store: KBStore) -> None:
    """A candidate lost to the window is attributed to `limit`, not the filters."""
    result = er.explain_ranking(store, query="jwt", limit=1)
    gates = {c["gate"] for c in result["candidates"]}
    assert "limit-dropped" in gates
    dropped = next(c for c in result["candidates"] if c["gate"] == "limit-dropped")
    assert _stages(dropped)[-1] == "strategy"


def test_budget_gate_reports_budget_dropped(store: KBStore) -> None:
    result = er.explain_ranking(store, query="jwt", limit=5, max_chars=1)
    assert result["retrieval"]["stages"]["budget"] == 1
    assert "budget-dropped" in {c["gate"] for c in result["candidates"]}


def test_scope_filtered_candidate_is_attributed_to_scope(store: KBStore) -> None:
    """A viewer-invisible claim dies at scope_filter, before the status gate."""
    store.put_claim(Claim(
        id="c-priv", text="jwt secret lives in vault",
        evidence=[store.list_sources()[0].id],
        scope=ArtifactScope(visibility=Visibility.PRIVATE, project="other-project"),
    ))
    health.rebuild_index(store)

    cand = _by_id(er.explain_ranking(
        store, query="jwt", limit=5, project="this-project"
    ))["c-priv"]
    assert cand["gate"] == "scope-filtered"
    assert _stages(cand) == ["hybrid"]


def test_scope_filtered_candidate_does_not_carry_its_summary(store: KBStore) -> None:
    """The gate that hid a candidate is reported; the text behind it is not.

    `kb.search` returns nothing for this viewer, so the same viewer asking
    `kb.explain_ranking` must not get the claim text back through the
    candidate's summary.
    """
    secret = "jwt signing secret is hunter2-example"
    store.put_claim(Claim(
        id="c-priv", text=secret,
        evidence=[store.list_sources()[0].id],
        scope=ArtifactScope(visibility=Visibility.PRIVATE, project="other-project"),
    ))
    health.rebuild_index(store)

    result = er.explain_ranking(store, query="jwt", limit=5, project="this-project")
    cand = _by_id(result)["c-priv"]

    assert cand["gate"] == "scope-filtered"
    assert cand["summary"] == ""
    assert "hunter2-example" not in json.dumps(result)

    # a candidate the viewer *can* retrieve still carries its summary
    assert _by_id(result)["c1"]["summary"]


def test_require_citations_names_the_uncited_claim(store: KBStore) -> None:
    """The uncited gate renames the responsible candidate, it does not drop it."""
    result = er.explain_ranking(
        store, query="jwt", limit=5, require_citations=True
    )
    assert result["retrieval"]["stages"]["require_citations"] is True
    # every fixture claim cites a source, so nothing is uncited
    assert {c["gate"] for c in result["candidates"]} == {"kept"}
    assert er._is_uncited(store, ("claim", "c1")) is False
    assert er._is_uncited(store, ("page", "p1")) is False


def test_uncited_gate_renames_a_surviving_claim(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The uncited branch is defensive — force it to pin the reported gate.

    ``Claim`` rejects ``evidence=[]`` on the model, so no stored claim can be
    uncited and this path cannot be reached through the public API. Forcing the
    predicate is the only way to assert the gate the branch would report if
    that invariant were ever relaxed.
    """
    monkeypatch.setattr(er, "_is_uncited", lambda _store, key: key[1] == "c1")
    result = er.explain_ranking(store, query="jwt", limit=5, require_citations=True)

    by_id = _by_id(result)
    assert by_id["c1"]["gate"] == "uncited"
    # the item is renamed, never dropped — its full stage chain is intact
    assert by_id["c1"]["stages"][-1]["stage"] == "limit"
    assert by_id["c2"]["gate"] == "kept"


def test_uncited_gate_is_not_applied_without_require_citations(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the flag the gate stays `kept` even for an uncited claim."""
    monkeypatch.setattr(er, "_is_uncited", lambda _store, _key: True)
    result = er.explain_ranking(store, query="jwt", limit=5)
    assert {c["gate"] for c in result["candidates"]} == {"kept"}


# --- stage reporting ------------------------------------------------------


def test_disabled_stages_are_reported_as_not_applied(store: KBStore) -> None:
    """A stage that is off still appears, flagged, so the chain reads whole."""
    _write_cfg(store, backend="hybrid", recency={"enabled": False})
    cand = _by_id(er.explain_ranking(store, query="jwt"))["c1"]
    recency = next(r for r in cand["stages"] if r["stage"] == "recency")
    assert recency["applied"] is False
    assert er.explain_ranking(store, query="jwt")["retrieval"]["stages"]["recency"] is (
        False
    )


def test_recency_reports_a_score_delta(store: KBStore) -> None:
    """Recency is rescoring-only, so its signal is the score delta."""
    _write_cfg(store, backend="hybrid", recency={"enabled": True, "half_life_days": 1})
    cand = _by_id(er.explain_ranking(store, query="jwt"))["c1"]
    recency = next(r for r in cand["stages"] if r["stage"] == "recency")
    assert recency["applied"] is True
    assert recency["score_delta"] is not None


def test_pages_first_stage_is_reported(store: KBStore) -> None:
    store.put_page(Page(id="p-live", title="jwt design", body="current",
                        type=PageType.CONCEPT))
    health.rebuild_index(store)
    _write_cfg(store, backend="hybrid", pages_first={"enabled": True, "boost": 2.0})

    cand = _by_id(er.explain_ranking(store, query="jwt", limit=5))["p-live"]
    pages_first = next(r for r in cand["stages"] if r["stage"] == "pages_first")
    assert pages_first["applied"] is True


def test_strategy_stage_reports_the_configured_plugin(store: KBStore) -> None:
    _write_cfg(store, backend="hybrid", strategy="vouch.strategies.provenance")
    result = er.explain_ranking(store, query="jwt")
    assert result["retrieval"]["stages"]["strategy"] == "vouch.strategies.provenance"
    strategy = next(
        r for r in _by_id(result)["c1"]["stages"] if r["stage"] == "strategy"
    )
    assert strategy["applied"] is True


def test_rerank_top_k_is_reported_only_when_rerank_is_on(store: KBStore) -> None:
    assert er.explain_ranking(store, query="jwt")["retrieval"]["stages"][
        "rerank_top_k"
    ] is None


# --- backend branches -----------------------------------------------------


def test_pinned_fts5_backend_is_explained(store: KBStore) -> None:
    _write_cfg(store, backend="fts5")
    result = er.explain_ranking(store, query="jwt")
    assert result["retrieval"]["used"] == "fts5"
    assert result["retrieval"]["stages"]["fusion"] is False
    assert _by_id(result)["c1"]["lexical_rank"] is not None


def test_pinned_embedding_backend_is_explained(store: KBStore) -> None:
    _write_cfg(store, backend="embedding")
    result = er.explain_ranking(store, query="jwt")
    assert result["retrieval"]["used"] == "embedding"


def test_pinned_substring_backend_is_explained(store: KBStore) -> None:
    _write_cfg(store, backend="substring")
    result = er.explain_ranking(store, query="jwt")
    assert result["retrieval"]["used"] == "substring"
    assert _by_id(result)["c1"]["gate"] == "kept"


def test_auto_falls_through_to_substring_when_retrievers_are_empty(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The substring fall-through is reported as the backend that served it."""
    from vouch import index_db

    monkeypatch.setattr(index_db, "search", lambda *a, **k: [])
    monkeypatch.setattr(index_db, "search_semantic", lambda *a, **k: [])
    result = er.explain_ranking(store, query="jwt")
    assert result["retrieval"]["used"] == "substring"


def test_lexical_sqlite_error_degrades_to_no_lexical_hits(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken fts5 index must not fault the explanation."""
    from vouch import index_db

    def _boom(*_a: object, **_k: object) -> list:
        raise sqlite3.Error("fts5 index is gone")

    monkeypatch.setattr(index_db, "search", _boom)
    result = er.explain_ranking(store, query="jwt")
    assert all(c["lexical_rank"] is None for c in result["candidates"])


def test_negative_limit_is_rejected(store: KBStore) -> None:
    with pytest.raises(ValueError, match="limit must be >= 0"):
        er.explain_ranking(store, query="jwt", limit=-1)


def test_viewer_is_echoed_back(store: KBStore) -> None:
    result = er.explain_ranking(store, query="jwt", project="proj-a", agent="agent-b")
    assert result["viewer"] == {"project": "proj-a", "agent": "agent-b"}


# --- surface parity -------------------------------------------------------


def test_jsonl_surface_serves_explain_ranking(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vouch import jsonl_server

    monkeypatch.setattr(jsonl_server, "_store", lambda: store)
    result = jsonl_server.HANDLERS["kb.explain_ranking"]({
        "query": "jwt", "limit": 5, "max_chars": 4000, "require_citations": False,
    })
    assert result["retrieval"]["stages"]["budget"] == 4000
    assert _by_id(result)["c1"]["gate"] == "kept"


def test_mcp_surface_serves_explain_ranking(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vouch import server

    monkeypatch.setattr(server, "_store", lambda: store)
    result = server.kb_explain_ranking("jwt", limit=5)
    assert _by_id(result)["c1"]["gate"] == "kept"


def test_explain_ranking_is_registered_in_capabilities() -> None:
    from vouch.capabilities import METHODS

    assert "kb.explain_ranking" in METHODS


def test_cli_text_output_names_stages_and_gates(store: KBStore) -> None:
    from click.testing import CliRunner

    from vouch.cli import cli

    store.put_claim(Claim(
        id="c-dead", text="jwt replaced by sessions",
        evidence=[store.list_sources()[0].id], status=ClaimStatus.SUPERSEDED,
    ))
    health.rebuild_index(store)
    _write_cfg(store, backend="hybrid", strategy="vouch.strategies.provenance",
               recency={"enabled": True, "half_life_days": 1})

    res = CliRunner().invoke(
        cli, ["explain-ranking", "jwt", "--limit", "2"]
    )
    assert res.exit_code == 0, res.output
    assert "backend:" in res.output
    assert "stages active:" in res.output
    assert "strategy=vouch.strategies.provenance" in res.output
    assert "gate=status-filtered" in res.output
    assert "(off)" in res.output


def test_cli_json_output_is_machine_readable(store: KBStore) -> None:
    import json

    from click.testing import CliRunner

    from vouch.cli import cli

    res = CliRunner().invoke(
        cli,
        ["explain-ranking", "jwt",
         "--format", "json", "--require-citations", "--max-chars", "4000"],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["query"] == "jwt"
    assert payload["retrieval"]["stages"]["require_citations"] is True


def test_cli_reports_no_active_stages_when_all_are_off(store: KBStore) -> None:
    from click.testing import CliRunner

    from vouch.cli import cli

    _write_cfg(store, backend="substring", recency={"enabled": False})
    res = CliRunner().invoke(
        cli, ["explain-ranking", "jwt"]
    )
    assert res.exit_code == 0, res.output
    assert "stages active: none" in res.output
