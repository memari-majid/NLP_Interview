#!/usr/bin/env python3
"""
Export all NLP problem/solution pairs into a CrowdAnki-compatible deck.

Output structure (created under ./anki/crowd_anki/NLP_Interview_Deck/):
- deck.json       : Deck metadata + note model definition
- notes.json      : Notes (cards) with Front/Back fields and tags
- media/          : Reserved for future media files (empty)

This script intentionally avoids non-standard dependencies so it can be run anywhere.

Usage:
  python scripts/export_to_crowdanki.py

Then in Anki (with CrowdAnki add-on):
  File -> CrowdAnki -> Import from disk -> select the NLP_Interview_Deck folder
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
NLP_DIR = REPO_ROOT / "NLP"
OUTPUT_DIR = REPO_ROOT / "anki" / "crowd_anki" / "NLP_Interview_Deck"
MEDIA_DIR = OUTPUT_DIR / "media"


def generate_stable_uuid(namespace_name: str, unique_name: str) -> str:
    """Generate a stable UUIDv5 for deterministic deck/model/note IDs.

    CrowdAnki expects UUIDs as strings. UUIDv5 gives stable IDs across runs given same inputs.
    """
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, namespace_name)
    return str(uuid.uuid5(namespace, unique_name))


def find_problem_solution_pairs(base_dir: Path) -> List[Tuple[Path, Path, str]]:
    """Return list of (problem_md, solution_py, tag) for each topic directory.

    The tag is derived from the immediate subdirectory name (e.g., TFIDF, Tokenization).
    """
    pairs: List[Tuple[Path, Path, str]] = []
    for topic_dir in sorted(base_dir.glob("*")):
        if not topic_dir.is_dir():
            continue
        problem_files = sorted(topic_dir.glob("*_problem.md"))
        solution_files = sorted(topic_dir.glob("*_solution.py"))
        if not problem_files or not solution_files:
            continue
        # Assume one problem/solution per directory (current repo design)
        pairs.append((problem_files[0], solution_files[0], topic_dir.name))
    return pairs


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"WARN: Failed to read {path}: {exc}")
        return ""


def extract_title_from_markdown(md_content: str, fallback: str) -> str:
    """Use first markdown header line as title; fallback to filename stem."""
    for line in md_content.splitlines():
        if line.strip().startswith("#"):
            # Strip leading #'s and whitespace
            return re.sub(r"^#+\s*", "", line).strip()
    return fallback


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def to_html_code_block(code: str, language: str = "") -> str:
    lang_class = f" class=\"language-{language}\"" if language else ""
    return f"<pre><code{lang_class}>" + html_escape(code) + "</code></pre>"


def build_front_back_html(problem_path: Path, solution_path: Path) -> Tuple[str, str]:
    problem_md = read_text(problem_path)
    solution_py = read_text(solution_path)

    title = extract_title_from_markdown(problem_md, fallback=problem_path.stem)
    # Keep problem concise; show full markdown after a horizontal rule
    front_parts: List[str] = [
        f"<h3>{html_escape(title)}</h3>",
        "<p><strong>Prompt:</strong></p>",
        f"<pre><code>{html_escape(problem_md.strip())}</code></pre>",
    ]
    front_html = "\n".join(front_parts)

    back_parts: List[str] = [
        "<p><strong>Solution (Python):</strong></p>",
        to_html_code_block(solution_py, language="python"),
        "<hr/>",
        "<p><em>Tip: Be explicit about shapes, complexity, and edge cases.</em></p>",
    ]
    back_html = "\n".join(back_parts)

    return front_html, back_html


def build_deck_json(deck_uuid: str, model_uuid: str, deck_name: str = "NLP Interview Deck") -> Dict:
    """Minimal deck.json compatible with CrowdAnki."""
    basic_note_model = {
        "crowdanki_uuid": model_uuid,
        "name": "Basic (NLP Code)",
        "type": 0,  # standard/basic
        "fields": [
            {"name": "Front"},
            {"name": "Back"},
        ],
        "templates": [
            {
                "name": "Card 1",
                "qfmt": "{{Front}}",
                "afmt": "{{Front}}<hr id=answer>{{Back}}",
            }
        ],
        "css": (
            ".card { font-family: -apple-system, Segoe UI, Roboto, sans-serif;"
            " font-size: 16px; color: #2d2d2d; background-color: #ffffff; }\n"
            "pre { background: #f6f8fa; padding: 10px; overflow: auto; }\n"
            "code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }"
        ),
    }

    deck_config_uuid = generate_stable_uuid("nlp-crowdanki-config", deck_uuid)

    deck_json = {
        "crowdanki_uuid": deck_uuid,
        "name": deck_name,
        "deck_configurations": [
            {
                "crowdanki_uuid": deck_config_uuid,
                "name": "Default",
                "new": {"perDay": 20},
                "rev": {"perDay": 200},
            }
        ],
        "deck_config_uuid": deck_config_uuid,
        "note_models": [basic_note_model],
        "children": [],
        "desc": "NLP coding interview Q/A generated from repository",
    }
    return deck_json


def build_note_entry(front_html: str, back_html: str, model_uuid: str, guid_source: str, tags: List[str]) -> Dict:
    # Use a stable 10-char GUID based on hash of content/path
    guid = hashlib.sha1(guid_source.encode("utf-8")).hexdigest()[:10]
    return {
        "note_model_uuid": model_uuid,
        "fields": [front_html, back_html],
        "guid": guid,
        "tags": tags,
    }


def write_crowdanki_files(deck_json: Dict, notes: List[Dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "deck.json").write_text(json.dumps(deck_json, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")


def main() -> int:
    if not NLP_DIR.exists():
        print(f"ERROR: NLP directory not found at {NLP_DIR}")
        return 1

    pairs = find_problem_solution_pairs(NLP_DIR)
    if not pairs:
        print("ERROR: No problem/solution pairs found.")
        return 1

    deck_uuid = generate_stable_uuid("nlp-crowdanki-deck", "NLP Interview Deck")
    model_uuid = generate_stable_uuid("nlp-crowdanki-model", "Basic (NLP Code)")
    deck_json = build_deck_json(deck_uuid=deck_uuid, model_uuid=model_uuid)

    notes: List[Dict] = []
    for problem_path, solution_path, tag in pairs:
        front_html, back_html = build_front_back_html(problem_path, solution_path)
        guid_source = str(problem_path.relative_to(REPO_ROOT))
        tags = ["NLP", tag, "Interview", "CrowdAnki"]
        notes.append(build_note_entry(front_html, back_html, model_uuid, guid_source, tags))

    write_crowdanki_files(deck_json, notes)

    print("CrowdAnki deck generated at:")
    print(f"  {OUTPUT_DIR}")
    print(f"  {OUTPUT_DIR / 'deck.json'}")
    print(f"  {OUTPUT_DIR / 'notes.json'}")
    print("Import into Anki via: File -> CrowdAnki -> Import from disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


