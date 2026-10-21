"""Cost and latency ledger: every model call is accounted for.

NFR-COST-01 asks for tokens, USD and latency for **every** model call; NFR-COST-02
asks that a price is a configuration edit, never a code edit. Hence:

* prices live in ``configs/pricing.yaml`` (:class:`PricingTable`), each row carrying
  ``retrieved_at`` and ``source`` — an unpriced model is reported, not silently free;
* :class:`CostLedger` keeps one :class:`CostRecord` per call with the token counts the
  provider reported (authoritative), plus the local estimate used only as a
  reconciliation diagnostic;
* :meth:`CostLedger.summary` aggregates per session, and :class:`CostReport` merges
  sessions into a report-level total (t7/t9 consume it; NFR-OBS-01).

The local token estimate is a coarse, deterministic heuristic by design: the 0.5%
reconciliation target of NFR-COST-01 is a release-gate measurement with a real
tokeniser (t11), not something an offline unit test may claim.

    # AT-COST-01  # AT-COST-02
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml

from answerdesk.contracts import Contract, PricingEntry
from answerdesk.errors import ConfigError

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PRICING_PATH = _REPO_ROOT / "configs" / "pricing.yaml"

#: Roughly 4 characters per token for alphabetic scripts; 1 token per CJK ideograph.
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def estimate_tokens(text: str) -> int:
    """Deterministic local token estimate (no model, no network, no dependency).

    Deliberately dumb and *documented as dumb*: it exists so the packer can enforce a
    budget and so the ledger can flag a provider that reports implausible usage. It is
    never used for billing — provider usage wins whenever it is available.
    """
    if not text or not text.strip():
        return 0
    cjk = len(_CJK.findall(text))
    rest = _CJK.sub(" ", text)
    return cjk + max(1, math.ceil(len(rest) / 4))


def estimate_messages_tokens(contents: Iterable[str]) -> int:
    """Token estimate for a whole prompt (message contents only)."""
    return sum(estimate_tokens(content) for content in contents)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, ``q`` in ``[0, 1]``; deterministic and exact for small n."""
    if not values:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be within [0, 1]")
    ordered = sorted(values)
    if q == 0.0:
        return ordered[0]
    rank = math.ceil(q * len(ordered))
    return ordered[min(max(rank, 1), len(ordered)) - 1]


# --------------------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------------------


def default_pricing_path() -> Path | None:
    """The checked-in pricing table, or ``None`` when the package is installed alone."""
    return _DEFAULT_PRICING_PATH if _DEFAULT_PRICING_PATH.is_file() else None


class PriceQuote(Contract):
    """The priced outcome of one call, with the provenance needed to defend the number."""

    usd: float
    priced: bool
    model: str
    pricing_source: str = ""


class PricingTable:
    """Pinned per-model prices, loaded from YAML with provenance (NFR-COST-02)."""

    def __init__(self, entries: Sequence[PricingEntry], *, source_path: Path | None = None) -> None:
        self.source_path = source_path
        self.entries: dict[str, PricingEntry] = {}
        for entry in entries:
            self.entries[entry.model] = entry

    # -- construction -------------------------------------------------------------
    @classmethod
    def from_entries(cls, entries: Sequence[PricingEntry]) -> PricingTable:
        return cls(entries)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PricingTable:
        file_path = Path(path)
        try:
            raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        except OSError as exc:  # pragma: no cover - filesystem dependent
            raise ConfigError(f"cannot read pricing table {file_path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"pricing table {file_path} is not valid YAML: {exc}") from exc
        if not isinstance(raw, dict) or "pricing" not in raw:
            raise ConfigError(f"pricing table {file_path} must contain a top-level 'pricing' list")
        rows = raw["pricing"]
        if not isinstance(rows, list):
            raise ConfigError(f"pricing table {file_path}: 'pricing' must be a list")
        try:
            entries = [PricingEntry.model_validate(row) for row in rows]
        except Exception as exc:  # pydantic ValidationError, kept message-bearing
            raise ConfigError(f"pricing table {file_path} has an invalid row: {exc}") from exc
        return cls(entries, source_path=file_path)

    @classmethod
    def default(cls) -> PricingTable:
        """The repository table when present, otherwise an empty (all-unpriced) table."""
        path = default_pricing_path()
        return cls.from_yaml(path) if path is not None else cls([])

    # -- lookup -------------------------------------------------------------------
    @staticmethod
    def candidates(model: str) -> list[str]:
        """Model-name candidates, so ``openai/gpt-4o-mini`` can hit a ``gpt-4o-mini`` row."""
        names = [model]
        if "/" in model:
            names.append(model.rsplit("/", 1)[-1])
        return names

    def entry_for(self, model: str) -> PricingEntry | None:
        for candidate in self.candidates(model):
            entry = self.entries.get(candidate)
            if entry is not None:
                return entry
        return None

    def price(self, model: str, *, input_tokens: int, output_tokens: int) -> PriceQuote:
        """USD for one call. An unknown model prices at 0 and is flagged ``priced=False``."""
        entry = self.entry_for(model)
        if entry is None:
            return PriceQuote(usd=0.0, priced=False, model=model, pricing_source="unknown")
        usd = (
            input_tokens / 1000.0 * entry.input_usd_per_1k_tokens
            + output_tokens / 1000.0 * entry.output_usd_per_1k_tokens
        )
        return PriceQuote(
            usd=round(usd, 10),
            priced=True,
            model=entry.model,
            pricing_source=f"{entry.source} @ {entry.retrieved_at}",
        )


# --------------------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------------------

UsageSource = Literal["provider", "estimated", "none"]


class CostRecord(Contract):
    """One model call, as the ledger remembers it (extra fields are a contract change)."""

    request_id: str
    session_id: str
    stage: str = "generation"
    provider: str = "http"
    preset: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_input_tokens: int | None = None
    usage_source: UsageSource = "provider"
    usd: float = 0.0
    priced: bool = True
    pricing_source: str = ""
    latency_ms: float = 0.0
    attempts: int = 1
    prompt_version: str = ""
    ok: bool = True
    error_code: str | None = None


class ModelCost(Contract):
    """Per-model aggregate (what the cost table in a report shows)."""

    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0


class CostSummary(Contract):
    """Session-level aggregate (NFR-COST-01: "aggregates per session/report")."""

    session_id: str
    calls: int = 0
    ok_calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd_total: float = 0.0
    usd_mean_per_call: float = 0.0
    usd_p95: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    attempts_total: int = 0
    unpriced_models: list[str] = []
    usage_sources: dict[str, int] = {}
    by_model: list[ModelCost] = []
    estimate_mismatch: float = 0.0

    def as_row(self) -> dict[str, Any]:
        """Flat dict for a markdown table or a JSONL report line."""
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd_total": round(self.usd_total, 8),
            "usd_mean_per_call": round(self.usd_mean_per_call, 8),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
        }


class CostReport(Contract):
    """Report-level aggregation across sessions (t7 writes it, t9 exposes it)."""

    sessions: list[CostSummary] = []
    usd_total: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    unpriced_models: list[str] = []

    @classmethod
    def from_summaries(cls, summaries: Sequence[CostSummary]) -> CostReport:
        models: set[str] = set()
        for summary in summaries:
            models.update(summary.unpriced_models)
        return cls(
            sessions=list(summaries),
            usd_total=round(sum(s.usd_total for s in summaries), 10),
            calls=sum(s.calls for s in summaries),
            input_tokens=sum(s.input_tokens for s in summaries),
            output_tokens=sum(s.output_tokens for s in summaries),
            unpriced_models=sorted(models),
        )


class CostLedger:
    """Append-only in-memory ledger for one session (t9 persists/bridges it)."""

    def __init__(
        self,
        *,
        session_id: str = "session",
        pricing: PricingTable | None = None,
        stage: str = "generation",
    ) -> None:
        self.session_id = session_id
        self.pricing = pricing if pricing is not None else PricingTable.default()
        self.stage = stage
        self.records: list[CostRecord] = []
        self._next_id = 1

    # -- recording ----------------------------------------------------------------
    def _mint_id(self) -> str:
        request_id = f"{self.session_id}-r{self._next_id:04d}"
        self._next_id += 1
        return request_id

    def record(
        self,
        *,
        preset: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        stage: str | None = None,
        provider: str = "http",
        prompt_version: str = "",
        estimated_input_tokens: int | None = None,
        usage_source: UsageSource = "provider",
        attempts: int = 1,
        ok: bool = True,
        error_code: str | None = None,
    ) -> CostRecord:
        """Price one call and append it. This is the *only* way a call gets accounted."""
        quote = self.pricing.price(model, input_tokens=input_tokens, output_tokens=output_tokens)
        record = CostRecord(
            request_id=self._mint_id(),
            session_id=self.session_id,
            stage=stage or self.stage,
            provider=provider,
            preset=preset,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_input_tokens=estimated_input_tokens,
            usage_source=usage_source,
            usd=quote.usd,
            priced=quote.priced,
            pricing_source=quote.pricing_source,
            latency_ms=round(latency_ms, 3),
            attempts=attempts,
            prompt_version=prompt_version,
            ok=ok,
            error_code=error_code,
        )
        self.records.append(record)
        return record

    def extend(self, records: Iterable[CostRecord]) -> None:
        """Import records produced elsewhere (e.g. a child ledger) without re-pricing."""
        for record in records:
            self.records.append(record)

    # -- aggregation --------------------------------------------------------------
    def summary(self) -> CostSummary:
        calls = len(self.records)
        latencies = [record.latency_ms for record in self.records]
        usds = [record.usd for record in self.records]
        mismatches = [
            abs(record.estimated_input_tokens - record.input_tokens) / record.input_tokens
            for record in self.records
            if record.estimated_input_tokens is not None and record.input_tokens > 0
        ]
        models: dict[str, list[CostRecord]] = {}
        for record in self.records:
            models.setdefault(record.model, []).append(record)
        by_model = [
            ModelCost(
                model=model,
                calls=len(rows),
                input_tokens=sum(r.input_tokens for r in rows),
                output_tokens=sum(r.output_tokens for r in rows),
                usd=round(sum(r.usd for r in rows), 10),
                latency_p50_ms=round(percentile([r.latency_ms for r in rows], 0.5), 3),
                latency_p95_ms=round(percentile([r.latency_ms for r in rows], 0.95), 3),
            )
            for model, rows in sorted(models.items())
        ]
        usage_sources: dict[str, int] = {}
        for record in self.records:
            usage_sources[record.usage_source] = usage_sources.get(record.usage_source, 0) + 1
        return CostSummary(
            session_id=self.session_id,
            calls=calls,
            ok_calls=sum(1 for record in self.records if record.ok),
            failed_calls=sum(1 for record in self.records if not record.ok),
            input_tokens=sum(record.input_tokens for record in self.records),
            output_tokens=sum(record.output_tokens for record in self.records),
            usd_total=round(sum(usds), 10),
            usd_mean_per_call=round(statistics.fmean(usds), 10) if usds else 0.0,
            usd_p95=round(percentile(usds, 0.95), 10),
            latency_p50_ms=round(percentile(latencies, 0.5), 3),
            latency_p95_ms=round(percentile(latencies, 0.95), 3),
            attempts_total=sum(record.attempts for record in self.records),
            unpriced_models=sorted({r.model for r in self.records if not r.priced}),
            usage_sources=usage_sources,
            by_model=by_model,
            estimate_mismatch=round(statistics.fmean(mismatches), 6) if mismatches else 0.0,
        )

    def to_json(self) -> str:
        """JSONL body: one JSON object per call plus the aggregate, for a report file."""
        payload = {
            "summary": self.summary().model_dump(mode="json"),
            "records": [record.model_dump(mode="json") for record in self.records],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    def reset(self) -> None:
        self.records.clear()
        self._next_id = 1

    def __len__(self) -> int:
        return len(self.records)


def summary_from_mapping(data: Mapping[str, Any]) -> CostSummary:
    """Validate a serialised summary (used by t7/t9 when reading a report back)."""
    return CostSummary.model_validate(dict(data))


__all__ = [
    "CostLedger",
    "CostRecord",
    "CostReport",
    "CostSummary",
    "ModelCost",
    "PriceQuote",
    "PricingTable",
    "UsageSource",
    "default_pricing_path",
    "estimate_messages_tokens",
    "estimate_tokens",
    "percentile",
    "summary_from_mapping",
]
