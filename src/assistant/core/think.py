"""
Split tagged spans out of a streamed reply.

Hosted providers return reasoning in a separate field, but local models served raw (Qwen, LFM,
DeepSeek distills on llama-server) emit ``<think>...</think>`` inline, and a model using a text
tool-call protocol emits its calls inline too. Tags arrive split across deltas ("<th", "ink>"), so the
splitter holds back any suffix that could still become a tag.
"""

from __future__ import annotations

OPEN = "<think>"
CLOSE = "</think>"


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``tag``."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


class TagSplitter:
    """Label text inside ``open``/``close`` as ``inside`` and everything else as ``outside``."""

    def __init__(self, open_tag: str, close_tag: str, inside: str, outside: str = "token") -> None:
        self._open = open_tag
        self._close = close_tag
        self._inside_kind = inside
        self._outside_kind = outside
        self._buf = ""
        self._inside = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Return ``(kind, text)`` pieces."""
        self._buf += text
        out: list[tuple[str, str]] = []
        while self._buf:
            tag = self._close if self._inside else self._open
            idx = self._buf.find(tag)
            if idx >= 0:
                self._emit(out, self._buf[:idx])
                self._buf = self._buf[idx + len(tag):]
                self._inside = not self._inside
                continue
            keep = _partial_suffix(self._buf, tag)
            self._emit(out, self._buf[: len(self._buf) - keep])
            self._buf = self._buf[len(self._buf) - keep:]
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        self._emit(out, self._buf)
        self._buf = ""
        return out

    def _emit(self, out: list[tuple[str, str]], text: str) -> None:
        if text:
            out.append((self._inside_kind if self._inside else self._outside_kind, text))


class ThinkSplitter(TagSplitter):
    def __init__(self) -> None:
        super().__init__(OPEN, CLOSE, "reasoning")
