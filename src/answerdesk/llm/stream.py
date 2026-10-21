"""A streamed completion that also satisfies the ``Sequence`` contract seam.

The frozen protocol ``ChatProvider.stream`` (``contracts.py``) returns
``Sequence[CompletionChunk]`` because most callers want the whole payload, but a real
SSE stream must be *incremental* — the service layer (t9) forwards tokens as they
arrive, and NFR-LAT-01 measures time-to-first-token. :class:`ChunkStream` is both:

* iterating pulls one chunk at a time from the underlying generator;
* ``len()``, indexing and slicing materialise the remaining chunks on demand.

A chunk that was already delivered is buffered, so a second pass over the stream is
free and deterministic.

    # AT-PORT-01
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import overload

from answerdesk.contracts import CompletionChunk


class ChunkStream(Sequence[CompletionChunk]):
    """Lazily materialised sequence over a chunk iterator."""

    def __init__(self, source: Iterator[CompletionChunk]) -> None:
        self._source: Iterator[CompletionChunk] | None = source
        self._buffer: list[CompletionChunk] = []

    # -- internal -----------------------------------------------------------------
    def _pull_one(self) -> bool:
        """Pull one chunk into the buffer; return False when the source is exhausted."""
        if self._source is None:
            return False
        try:
            chunk = next(self._source)
        except StopIteration:
            self._source = None
            return False
        self._buffer.append(chunk)
        return True

    def _fill_to(self, index: int) -> None:
        while len(self._buffer) <= index:
            if not self._pull_one():
                break

    def _fill_all(self) -> None:
        while self._pull_one():
            pass

    # -- helpers ------------------------------------------------------------------
    def materialise(self) -> list[CompletionChunk]:
        """All chunks as a list (also closes the source)."""
        self._fill_all()
        return list(self._buffer)

    def text(self) -> str:
        """Concatenated deltas — the answer text, without buffering by the caller."""
        return "".join(chunk.delta for chunk in self.materialise())

    @property
    def started(self) -> bool:
        """True once at least one chunk (or the first exhaustion) has been observed."""
        return bool(self._buffer) or self._source is None

    # -- Sequence protocol --------------------------------------------------------
    def __iter__(self) -> Iterator[CompletionChunk]:
        index = 0
        while True:
            if index >= len(self._buffer) and not self._pull_one():
                return
            yield self._buffer[index]
            index += 1

    @overload
    def __getitem__(self, index: int) -> CompletionChunk: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[CompletionChunk]: ...

    def __getitem__(self, index: int | slice) -> CompletionChunk | Sequence[CompletionChunk]:
        if isinstance(index, slice):
            self._fill_all()
            return self._buffer[index]
        if index < 0:
            self._fill_all()
            return self._buffer[index]
        self._fill_to(index)
        return self._buffer[index]

    def __len__(self) -> int:
        self._fill_all()
        return len(self._buffer)


__all__ = ["ChunkStream"]
