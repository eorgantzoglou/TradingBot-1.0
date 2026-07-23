"""Durable output: turn a ResearchReport into files you can keep.

`scout research` prints a memo to the terminal, but a research tool should leave
an artifact you can revisit, diff over time, and cite. This package writes each
report twice -- Markdown to read, JSON to machine-process -- into a day-stamped
directory, so a watchlist's analysis accumulates as a browsable record.
"""

from scout.output.report import render_markdown, report_to_dict, write_reports

__all__ = ["render_markdown", "report_to_dict", "write_reports"]
