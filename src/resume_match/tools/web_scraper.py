"""Deprecated compatibility shim; use :mod:`resume_match.adapters.web_scraper`."""

from resume_match.adapters.web_scraper import (
    _clean_html_to_text,
    fetch_and_extract,
    fetch_with_playwright,
)

__all__ = ["_clean_html_to_text", "fetch_and_extract", "fetch_with_playwright"]
