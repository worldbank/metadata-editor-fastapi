"""
Generic assistant core: LLM gateway, prompt assembly, streaming protocol.

Nothing in this package may import from ``src.assistant.host``, ``main`` or any other editor module;
it is extracted unchanged into portable-slm later. ``test_core_isolation`` enforces this.
"""

PROTOCOL = "pslm-server/1"
