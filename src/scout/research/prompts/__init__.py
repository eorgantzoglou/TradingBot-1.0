"""Prompt text for the research pipeline, kept apart from the call logic.

Prompts are data, not code: they change for different reasons (wording, tone,
red-flag taxonomy) and on a different cadence than the harness plumbing that
sends them. Isolating them also keeps the stable, cache-friendly system prompt
in one obvious place -- the prefix the provider caches must not drift on an
unrelated edit.
"""
