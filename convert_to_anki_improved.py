#!/usr/bin/env python3
"""
Generate an Anki CrowdAnki JSON deck with precise, interview-ready NLP flashcards.

Design goals:
- Atomic cards (one idea per card) for better memorization
- Precise prompts with minimal, high-signal answers
- Works directly with CrowdAnki (Anki 25.x schema: fields include sticky/rtl/font/size)
- Produces both a root JSON (for GitHub import) and a folder JSON (for disk import)

Usage:
  python convert_to_anki_improved.py

Outputs:
  - NLP_Interview.json
  - anki_deck_improved/NLP_Interview_Improved.json
"""

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).parent
NLP_DIR = ROOT / "NLP"
OUT_DIR = ROOT / "anki_deck_improved"


def md5_short(text: str, length: int = 13) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def extract_title(markdown_text: str, fallback: str) -> str:
    # First non-empty H1/H2 line, else fallback
    for line in markdown_text.splitlines():
        line = line.strip()
        if re.match(r"^#{1,2}\s+", line):
            return re.sub(r"^#{1,2}\s+", "", line).strip()
    return fallback


def extract_problem_summary(markdown_text: str, max_chars: int = 160) -> str:
    # First 2-3 non-heading lines stitched; strip code fences
    clean = re.sub(r"```[\s\S]*?```", "", markdown_text)
    lines = [ln.strip() for ln in clean.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    summary = " ".join(lines[:3])
    return (summary[:max_chars] + ("…" if len(summary) > max_chars else "")) or "What is the key idea?"


def extract_enumerated_steps(text: str, limit: int = 6) -> List[str]:
    steps = []
    for line in text.splitlines():
        ln = line.strip()
        if re.match(r"^(?:\d+\.|[-•])\s+", ln) or ln.upper().startswith("STEP "):
            ln = re.sub(r"^(?:\d+\.|[-•])\s+", "", ln)
            steps.append(ln)
        if len(steps) >= limit:
            break
    return steps


def extract_complexities(text: str) -> Tuple[str, str]:
    # Try to find explicit mentions first
    time_match = re.search(r"(?i)(time).*?O\([^)]*\)", text)
    space_match = re.search(r"(?i)(space).*?O\([^)]*\)", text)
    time = time_match.group(0) if time_match else ""
    space = space_match.group(0) if space_match else ""
    return time, space


def extract_formulas(topic: str, text: str) -> List[str]:
    formulas: List[str] = []
    # Heuristics for common formulas
    patterns = [
        r"TF-?IDF.*?=|IDF\s*=\s*log\s*\(",
        r"cosine.*?=\s*.*?/[\s\S]*?\)",
        r"Attention\s*\(Q,K,V\)\s*=\s*softmax\(QK\^T\s*/\s*√?d_k\)V",
        r"softmax\s*\(x\)\s*=",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            # Include a canonical version when possible
            if "Attention" in pat:
                formulas.append("Attention(Q,K,V) = softmax(QK^T / √d_k) V")
            elif "cosine" in pat.lower():
                formulas.append("cosine_sim(A,B) = (A·B) / (||A|| × ||B||)")
            elif "TF-IDF" in pat or "IDF" in pat:
                formulas.append("TF-IDF(t,d) = TF(t,d) × IDF(t); IDF(t)=log(N/df(t))")
            else:
                formulas.append(m.group(0))
    # De-duplicate
    uniq: List[str] = []
    for f in formulas:
        if f not in uniq:
            uniq.append(f)
    return uniq[:2]


def ast_functions(code: str, max_functions: int = 2, max_lines: int = 18) -> List[Tuple[str, str]]:
    """Return list of (name, source_snippet) for top-level functions, trimmed."""
    results: List[Tuple[str, str]] = []
    try:
        tree = ast.parse(code)
        lines = code.splitlines()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                start = node.lineno - 1
                end = getattr(node, "end_lineno", start + max_lines)  # py>=3.8
                snippet_lines = lines[start:end]
                # Trim long functions
                snippet_lines = snippet_lines[:max_lines]
                snippet = "\n".join(snippet_lines)
                results.append((node.name, snippet))
                if len(results) >= max_functions:
                    break
    except Exception:
        pass
    return results


def build_note(front: str, back: str, topic: str, type_name: str, model_uuid: str) -> Dict:
    return {
        "__type__": "Note",
        "fields": [front, back, topic, type_name],
        "guid": f"nlp_{md5_short(front + type_name)}",
        "note_model_uuid": model_uuid,
        "tags": [topic.lower().replace(" ", "_"), type_name]
    }


def note_model(model_uuid: str, model_name: str) -> Dict:
    return {
        "__type__": "NoteModel",
        "crowdanki_uuid": model_uuid,
        "sortf": 0,
        "latexPre": "\\documentclass[12pt]{article}\n\\special{papersize=3in,5in}\n\\usepackage{amssymb,amsmath}\n\\pagestyle{empty}\n\\setlength{\\parindent}{0in}\n\\begin{document}\n",
        "latexPost": "\\end{document}",
        "tags": [],
        "vers": [],
        "css": (
            "\n.card {\n"
            "    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;\n"
            "    font-size: 16px;\n"
            "    text-align: left;\n"
            "    color: #2d3748;\n"
            "    background-color: #ffffff;\n"
            "    padding: 16px;\n"
            "}\npre {\n"
            "    background-color: #f7fafc;\n"
            "    color: #2d3748;\n"
            "    padding: 12px;\n"
            "    border-radius: 8px;\n"
            "    overflow-x: auto;\n"
            "    font-size: 14px;\n"
            "    line-height: 1.5;\n"
            "    font-family: 'SF Mono', Monaco, Consolas, monospace;\n"
            "    border-left: 4px solid #667eea;\n"
            "}\ncode {\n"
            "    background-color: #edf2f7;\n"
            "    padding: 2px 6px;\n"
            "    border-radius: 4px;\n"
            "    color: #d6336c;\n"
            "    font-size: 14px;\n"
            "}\n"
        ),
        "flds": [
            {"name": "Front", "ord": 0, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""},
            {"name": "Back",  "ord": 1, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""},
            {"name": "Topic", "ord": 2, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""},
            {"name": "Type",  "ord": 3, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": ""}
        ],
        "name": model_name,
        "tmpls": [{
            "afmt": "{{FrontSide}}<hr id=answer>{{Back}}<br><br><small style='color:#718096'>{{Topic}} • {{Type}}</small>",
            "bqfmt": "",
            "did": None,
            "name": "Card 1",
            "ord": 0,
            "qfmt": "{{Front}}"
        }],
        "type": 0
    }


def create_deck_skeleton(name: str, deck_uuid: str, config_uuid: str, model: Dict) -> Dict:
    return {
        "__type__": "Deck",
        "children": [],
        "crowdanki_uuid": deck_uuid,
        "deck_config_uuid": config_uuid,
        "deck_configurations": [{
            "__type__": "DeckConfig",
            "autoplay": True,
            "crowdanki_uuid": config_uuid,
            "dyn": False,
            "name": name,
            "new": {"delays": [1, 10], "initialFactor": 2500, "ints": [1, 4, 7], "order": 1, "perDay": 30},
            "rev": {"ease4": 1.3, "hardFactor": 1.2, "ivlFct": 1.0, "maxIvl": 36500, "perDay": 100}
        }],
        "desc": "Interview-ready NLP cards (atomic, precise, mobile friendly).",
        "dyn": 0,
        "extendNew": 10,
        "extendRev": 50,
        "media_files": [],
        "name": name,
        "note_models": [model],
        "notes": []
    }


def build_cards_for_pair(problem_path: Path, solution_path: Path, topic: str, model_uuid: str) -> List[Dict]:
    cards: List[Dict] = []
    problem_md = read_text(problem_path)
    solution_py = read_text(solution_path)

    title = extract_title(problem_md, problem_path.stem.replace("_", " "))
    summary = extract_problem_summary(problem_md)

    # Understanding card (front: question, back: concise summary)
    front = f"<b>{title}</b><br>What is the key idea?"
    back = summary
    cards.append(build_note(front, back, topic, "understanding", model_uuid))

    # Implementation cards: top 1-2 concise functions
    for fname, snippet in ast_functions(solution_py, max_functions=2, max_lines=18):
        impl_front = f"{title}<br>Implement: <code>{fname}()</code>"
        impl_back = f"<pre><code>{snippet}</code></pre>"
        cards.append(build_note(impl_front, impl_back, topic, "implementation", model_uuid))

    # Formula card(s)
    for f in extract_formulas(topic, solution_py + "\n" + problem_md):
        formula_front = f"{title}<br>Write the formula"
        formula_back = f"<pre>{f}</pre>"
        cards.append(build_note(formula_front, formula_back, topic, "formula", model_uuid))

    # Steps card
    steps = extract_enumerated_steps(problem_md + "\n" + solution_py)
    if steps:
        steps_front = f"{title}<br>List the algorithm steps"
        steps_back = "<pre>" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) + "</pre>"
        cards.append(build_note(steps_front, steps_back, topic, "steps", model_uuid))

    # Complexity card
    time_c, space_c = extract_complexities(solution_py)
    complexity_lines = []
    if time_c:
        complexity_lines.append(time_c)
    if space_c:
        complexity_lines.append(space_c)
    if complexity_lines:
        comp_front = f"{title}<br>What's the complexity?"
        comp_back = "<pre>" + "\n".join(complexity_lines) + "</pre>"
        cards.append(build_note(comp_front, comp_back, topic, "complexity", model_uuid))

    # Pitfalls/Edge cases (from ❌ lines or 'Edge:' comments)
    pitfalls = []
    for ln in solution_py.splitlines():
        s = ln.strip()
        if s.startswith("❌") or re.search(r"(?i)(edge|pitfall|gotcha)", s):
            pitfalls.append(re.sub(r"^[-•]\s*", "", s))
        if len(pitfalls) >= 4:
            break
    if pitfalls:
        pit_front = f"{title}<br>Common pitfalls?"
        pit_back = "<pre>" + "\n".join(pitfalls) + "</pre>"
        cards.append(build_note(pit_front, pit_back, topic, "pitfalls", model_uuid))

    return cards


def gather_pairs() -> List[Tuple[Path, Path, str]]:
    pairs: List[Tuple[Path, Path, str]] = []
    for topic_dir in sorted(NLP_DIR.glob("*/")):
        if not topic_dir.is_dir():
            continue
        topic_name = topic_dir.name.replace("_", " ")
        for problem in sorted(topic_dir.glob("*_problem.md")):
            solution = problem.with_suffix('.py').with_name(problem.stem.replace("_problem", "_solution") + ".py")
            if solution.exists():
                pairs.append((problem, solution, topic_name))
    return pairs


def main() -> None:
    print("Building improved CrowdAnki deck from NLP problems…")
    pairs = gather_pairs()
    print(f"Found {len(pairs)} problem/solution pairs")

    model_uuid = "nlp-model-improved"
    model = note_model(model_uuid, "NLP Interview Card (Improved)")
    deck = create_deck_skeleton("NLP Interview Prep (Improved)", "nlp-interview-deck-improved-2024", "nlp-deck-config-improved", model)

    notes: List[Dict] = []
    for problem, solution, topic in pairs:
        notes.extend(build_cards_for_pair(problem, solution, topic, model_uuid))

    deck["notes"] = notes

    # Ensure output directory exists
    OUT_DIR.mkdir(exist_ok=True)

    # Write folder JSON
    out_path_folder = OUT_DIR / "NLP_Interview_Improved.json"
    out_path_folder.write_text(json.dumps(deck, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also write root JSON for GitHub import
    out_path_root = ROOT / "NLP_Interview.json"
    out_path_root.write_text(json.dumps(deck, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Wrote {out_path_folder} ({len(notes)} notes)")
    print(f"✅ Wrote {out_path_root}   ({len(notes)} notes)")
    print("Import via CrowdAnki: Import from disk (folder) or Import from GitHub (root JSON present).")


if __name__ == "__main__":
    main()


