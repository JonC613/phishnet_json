"""Utilities for working with Phish show data and retrieval."""

from importlib import metadata


def get_version() -> str:
    """Return the installed package version if available."""
    try:
        return metadata.version("phishbot")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__all__ = ["get_version"]
