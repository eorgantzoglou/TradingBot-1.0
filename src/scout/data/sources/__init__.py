"""Primary filing sources.

Each module here wraps one publisher's API and yields raw, unmodified payloads.
No parsing and no normalization happens at this layer -- that is the whole point
of the archive (see `scout.data.archive`).
"""

from scout.data.sources.base import DocumentRef, RawDocument, Source

__all__ = ["DocumentRef", "RawDocument", "Source"]
