"""Helpers for loading and formatting Phish show data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from langchain.schema import Document


def load_shows(shows_dir: str | Path, limit: int | None = None) -> List[Dict]:
    """Load show JSON files from the provided directory.

    The loader understands both single-show JSON files and yearly collections
    that contain a list of shows. Files are sorted alphabetically to keep the
    output deterministic, and shows are sorted by showdate when possible.
    """

    directory = Path(shows_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Show directory not found: {shows_dir}")

    shows: List[Dict] = []
    for path in sorted(directory.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, list):
            shows.extend(show for show in data if isinstance(show, dict))
        elif isinstance(data, dict):
            shows.append(data)

        if limit and len(shows) >= limit:
            shows = shows[:limit]
            break

    shows.sort(key=lambda show: show.get("showdate", ""))
    return shows


def _location_string(show: Dict) -> str:
    pieces = [show.get("city"), show.get("state"), show.get("country")]
    return ", ".join([piece for piece in pieces if piece])


def format_show_text(show: Dict) -> str:
    """Create a human-friendly multi-line description for a show."""

    lines: List[str] = []
    lines.append(f"Date: {show.get('showdate', 'Unknown date')}")
    lines.append(f"Venue: {show.get('venue', 'Unknown venue')}")
    location = _location_string(show)
    if location:
        lines.append(f"Location: {location}")
    if tour := show.get("tour_name"):
        lines.append(f"Tour: {tour}")

    sets = show.get("sets", {}) or {}
    if sets:
        for set_name in sorted(sets.keys()):
            set_label = _format_set_label(set_name)
            songs = sets.get(set_name, []) or []
            song_chunks = []
            for song in songs:
                title = song.get("song", "")
                if song.get("jam"):
                    title += " [jam]"
                if song.get("transition"):
                    title += " >"
                if foot := song.get("footnote"):
                    title += f" ({foot})"
                song_chunks.append(title.strip())
            joined_songs = " ".join(chunk for chunk in song_chunks if chunk)
            lines.append(f"{set_label}: {joined_songs}")

    if notes := show.get("setlist_notes"):
        sanitized = notes.replace("<p>", " ").replace("</p>", " ")
        sanitized = sanitized.replace("<br>", " ").replace("\n", " ")
        lines.append(f"Notes: {sanitized.strip()}")

    return "\n".join(line for line in lines if line)


def _format_set_label(name: str) -> str:
    lowered = name.lower()
    if lowered == "e":
        return "Encore"
    if lowered.startswith("e") and lowered != "e":
        return f"Encore {name.upper()}"
    return f"Set {name}"


def build_documents(shows: Iterable[Dict]) -> List[Document]:
    """Convert show dictionaries into LangChain documents."""

    documents: List[Document] = []
    for show in shows:
        metadata = {
            "showdate": show.get("showdate"),
            "venue": show.get("venue"),
            "city": show.get("city"),
            "state": show.get("state"),
            "country": show.get("country"),
            "tour_name": show.get("tour_name"),
            "source": show.get("showdate"),
        }
        documents.append(Document(page_content=format_show_text(show), metadata=metadata))
    return documents


def summarize_shows(shows: Iterable[Dict]) -> Dict:
    """Produce simple dataset statistics for validation output."""

    shows_list = list(shows)
    stats = {
        "total_shows": len(shows_list),
        "shows_with_venue": 0,
        "shows_with_sets": 0,
        "total_songs": 0,
        "years": {},
        "unique_venues": set(),
    }

    for show in shows_list:
        date = show.get("showdate")
        venue = show.get("venue")
        if venue:
            stats["shows_with_venue"] += 1
            stats["unique_venues"].add(venue)
        if date and len(date) >= 4:
            year = date[:4]
            stats["years"][year] = stats["years"].get(year, 0) + 1
        sets = show.get("sets", {}) or {}
        if sets:
            stats["shows_with_sets"] += 1
            for songs in sets.values():
                stats["total_songs"] += len(songs or [])

    stats["unique_venues"] = len(stats["unique_venues"])
    return stats
