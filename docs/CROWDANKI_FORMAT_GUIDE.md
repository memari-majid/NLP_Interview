# CrowdAnki Format Guide

## Overview

This guide provides the definitive format for creating CrowdAnki-compatible JSON files for Anki flashcard import. All JSON files in this repository follow this exact structure to ensure error-free imports.

## Required Structure

### Complete JSON Template

```json
{
  "__type__": "Deck",
  "children": [],
  "crowdanki_uuid": "deck-[unique-id]",
  "deck_config_uuid": "default-config",
  "deck_configurations": [
    {
      "__type__": "DeckConfig",
      "crowdanki_uuid": "default-config",
      "name": "Default",
      "autoplay": true,
      "dyn": false,
      "lapse": {
        "delays": [10],
        "leechAction": 0,
        "leechFails": 8,
        "minInt": 1,
        "mult": 0
      },
      "maxTaken": 60,
      "new": {
        "bury": false,
        "delays": [1, 10],
        "initialFactor": 2500,
        "ints": [1, 4, 0],
        "order": 1,
        "perDay": 20
      },
      "replayq": true,
      "rev": {
        "bury": false,
        "ease4": 1.3,
        "hardFactor": 1.2,
        "ivlFct": 1,
        "maxIvl": 36500,
        "perDay": 200
      },
      "timer": 0
    }
  ],
  "desc": "Comprehensive flashcards for [Chapter Name]",
  "dyn": false,
  "extendNew": 10,
  "extendRev": 50,
  "media_files": [],
  "name": "Chapter Name",
  "note_models": [
    {
      "__type__": "NoteModel",
      "crowdanki_uuid": "ml-nlp-interview-model",
      "css": ".card {\n font-family: arial;\n font-size: 20px;\n text-align: center;\n color: black;\n background-color: white;\n}\n\n.front {\n font-weight: bold;\n color: #2c3e50;\n}\n\n.back {\n text-align: left;\n padding: 20px;\n}\n\n.concept {\n font-weight: bold;\n color: #e74c3c;\n margin-bottom: 10px;\n}\n\n.intuition {\n color: #3498db;\n font-style: italic;\n margin-bottom: 10px;\n}\n\n.mechanics {\n color: #27ae60;\n margin-bottom: 10px;\n}\n\n.tradeoffs {\n color: #f39c12;\n margin-bottom: 10px;\n}\n\n.applications {\n color: #9b59b6;\n margin-bottom: 10px;\n}\n\n.memory-hook {\n background-color: #ecf0f1;\n padding: 10px;\n border-left: 4px solid #34495e;\n font-style: italic;\n color: #34495e;\n}",
      "flds": [
        {
          "__type__": "NoteModelField",
          "font": "Arial",
          "media": [],
          "name": "Front",
          "ord": 0,
          "rtl": false,
          "size": 20,
          "sticky": false
        },
        {
          "__type__": "NoteModelField",
          "font": "Arial",
          "media": [],
          "name": "Back",
          "ord": 1,
          "rtl": false,
          "size": 20,
          "sticky": false
        },
        {
          "__type__": "NoteModelField",
          "font": "Arial",
          "media": [],
          "name": "Tags",
          "ord": 2,
          "rtl": false,
          "size": 20,
          "sticky": false
        },
        {
          "__type__": "NoteModelField",
          "font": "Arial",
          "media": [],
          "name": "Difficulty",
          "ord": 3,
          "rtl": false,
          "size": 20,
          "sticky": false
        }
      ],
      "latexPost": "\\end{document}",
      "latexPre": "\\documentclass[12pt]{article}\n\\special{papersize=3in,5in}\n\\usepackage[utf8]{inputenc}\n\\usepackage{amssymb,amsmath}\n\\pagestyle{empty}\n\\setlength{\\parindent}{0in}\n\\begin{document}\n",
      "name": "ML/NLP Interview",
      "req": [
        [0, "all"]
      ],
      "sortf": 0,
      "tags": [],
      "tmpls": [
        {
          "__type__": "CardTemplate",
          "afmt": "{{FrontSide}}\n\n<hr id=answer>\n\n<div class=\"back\">\n{{Back}}\n</div>",
          "bafmt": "",
          "bqfmt": "",
          "did": null,
          "name": "Card 1",
          "ord": 0,
          "qfmt": "<div class=\"front\">{{Front}}</div>"
        }
      ],
      "type": 0
    }
  ],
  "notes": [
    {
      "__type__": "Note",
      "crowdanki_uuid": "note-[unique-id-1]",
      "fields": [
        "Interview-style question?",
        "<div class=\"concept\"><strong>Concept:</strong> ...</div><br><br><div class=\"intuition\"><strong>Intuition:</strong> ...</div><br><br><div class=\"mechanics\"><strong>Mechanics:</strong> ...</div><br><br><div class=\"tradeoffs\"><strong>Trade-offs:</strong> ...</div><br><br><div class=\"applications\"><strong>Applications:</strong> ...</div><br><br><div class=\"memory-hook\"><strong>Memory Hook:</strong> ...</div>",
        "ML Intuition Medium",
        "Medium"
      ],
      "flags": 0,
      "guid": "guid-[unique-id-1]",
      "note_model_uuid": "ml-nlp-interview-model",
      "tags": ["ML", "Intuition", "Medium"]
    }
  ]
}
```

## Critical Requirements

### Required Keys (All Must Be Present)

1. **`"__type__": "Deck"`** - Main deck identifier (MUST be first)
2. **`"children": []`** - Subdeck array (usually empty)
3. **`"crowdanki_uuid"`** - Unique deck identifier
4. **`"deck_config_uuid"`** - Reference to deck configuration
5. **`"deck_configurations"`** - Spaced repetition settings
6. **`"desc"`** - Deck description
7. **`"dyn"`** - Dynamic deck flag (usually false)
8. **`"extendNew"`** - New card extension limit
9. **`"extendRev"`** - Review card extension limit
10. **`"media_files"`** - Media file references (usually empty)
11. **`"name"`** - Deck display name
12. **`"note_models"`** - Note type definitions
13. **`"notes"`** - Flashcard data

### Required `__type__` Annotations

Every major object MUST include a `__type__` field:

- **Deck**: `"__type__": "Deck"`
- **Deck Config**: `"__type__": "DeckConfig"`
- **Note Model**: `"__type__": "NoteModel"`
- **Note Model Field**: `"__type__": "NoteModelField"`
- **Card Template**: `"__type__": "CardTemplate"`
- **Note**: `"__type__": "Note"`

## Common Import Errors

### `KeyError: 'children'`
- **Cause**: Missing `children` key
- **Fix**: Add `"children": []` to deck root

### `KeyError: 'note_models'`
- **Cause**: Missing `note_models` key
- **Fix**: Add complete `note_models` array with all required fields

### `KeyError: '__type__'`
- **Cause**: Missing `__type__` annotations
- **Fix**: Add appropriate `__type__` to all objects

## Field Definitions

### Front Field (Question)
- Contains the interview-style question
- Plain text format
- Should be concise and clear

### Back Field (Answer)
- Contains structured HTML answer with styled sections
- Format: `<div class="concept"><strong>Concept:</strong> ...</div>`
- Sections: concept, intuition, mechanics, tradeoffs, applications, memory-hook

### Tags Field
- Space-separated tags for filtering
- Example: `"ML Intuition Medium"`

### Difficulty Field
- Single difficulty level
- Values: `"Easy"`, `"Medium"`, `"Hard"`

## CSS Styling

The included CSS provides:
- **Mobile-optimized** layout
- **Color-coded sections**:
  - 🔴 Concept (Red)
  - 🔵 Intuition (Blue)  
  - 🟢 Mechanics (Green)
  - 🟠 Trade-offs (Orange)
  - 🟣 Applications (Purple)
  - ⚫ Memory Hook (Gray box)

## File Organization

### Directory Structure
```
flashcards/
├── Natural Language Processing in Action/
│   ├── 01_nlp_overview/
│   │   └── 01_nlp_overview.json
│   ├── 02_tokenization/
│   │   └── 02_tokenization.json
│   └── ...
├── nlp/
│   ├── NLP_Fundamentals/
│   └── ...
└── ml/
    ├── ML_Fundamentals/
    └── ...
```

### Naming Convention
- **Directory**: `[number]_[short_name]` (e.g., `01_nlp_overview`)
- **JSON File**: `[directory_name].json` (e.g., `01_nlp_overview.json`)

## Import Process

1. **Open Anki** with CrowdAnki add-on installed
2. **Go to File → Import**
3. **Select directory** containing the JSON file
4. **CrowdAnki auto-detects** the format
5. **Import completes** without errors

## Validation Checklist

Before importing, verify:
- [ ] All required keys present
- [ ] All `__type__` annotations included
- [ ] JSON syntax valid (use JSONLint)
- [ ] Unique IDs used throughout
- [ ] HTML formatting correct in answers
- [ ] Directory and filename match

## Troubleshooting

### Import Fails
1. Check for missing `__type__` fields
2. Validate JSON syntax
3. Ensure all required keys present
4. Compare with working examples in this repo

### Cards Don't Display Properly
1. Check HTML formatting in Back field
2. Verify CSS is included in note model
3. Test with simple text first

This format has been tested and works perfectly with CrowdAnki for error-free Anki imports.
