# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview
This is a comprehensive NLP interview preparation system with 27 core problems covering classical NLP to modern LLMs. The repository is designed for efficient learning through bite-sized problems, Anki cards, and company-specific guides. The repository is optimized for Anki flashcard generation with 123+ cards across 6 different card types per problem.

## Common Commands

### Running Solutions
```bash
# Run any solution directly (they include test cases)
python NLP/TFIDF/tfidf_solution.py
python NLP/Attention_Mechanisms/self_attention_solution.py
python NLP/Text_Classification/text_classification_solution.py
```

### Interactive Tools
```bash
# Find problems by difficulty, company, or topic
python scripts/problem_finder.py

# Generate Anki flashcards (123+ bite-sized cards)
python scripts/convert_to_anki_optimized.py

# Generate standard Anki deck (30+ cards)
python scripts/convert_to_anki.py

# Clean and organize repository structure
python scripts/clean_and_organize.py
```

## Architecture & Structure

### Problem Organization
Each topic in `NLP/` contains:
- `*_problem.md`: Problem statement with examples and constraints
- `*_solution.py`: Heavily commented solution with test cases
- Solutions are self-contained with inline tests (no external testing framework)

### Key Components
1. **Problem Solutions**: Each solution file is standalone with:
   - Detailed docstrings explaining the approach
   - Step-by-step implementation with interview-focused comments
   - Edge case handling
   - Complexity analysis
   - Built-in test cases at the bottom

2. **Problem Finder** (`scripts/problem_finder.py`): Interactive navigation system with problem metadata including difficulty, time estimates, companies, and related problems.

3. **Anki Converters**: Transform problems into flashcards
   - `scripts/convert_to_anki_optimized.py`: Creates 6 card types per problem (understanding, implementation, formulas, complexity, edge cases, insights)
   - `scripts/convert_to_anki.py`: Creates standard deck with full solutions
   - Cards are bite-sized (5-15 lines) for mobile learning

### Solution Pattern
Solutions follow a consistent interview-ready format:
```python
def solution_function(params):
    """Docstring with approach explanation"""
    # STEP 1: Handle edge cases
    # STEP 2: Core algorithm with interview talking points
    # Complexity comments inline
    # Return result

# Test cases at bottom
if __name__ == "__main__":
    # Run tests
```

## Development Guidelines

### When Modifying Solutions
- Maintain the interview-focused comment style with STEP markers
- Include complexity analysis as comments
- Keep edge case handling explicit
- Ensure test cases run successfully

### When Adding New Problems
- Follow the existing structure: `topic_name_problem.md` and `topic_name_solution.py`
- Include in `scripts/problem_finder.py` PROBLEMS dictionary with metadata
- Add interview talking points as comments prefixed with "Key:", "Interview:", etc.

### Testing Approach
- Solutions are self-testing via `if __name__ == "__main__"` blocks
- No external testing framework is used
- Each solution prints test results when run directly

## Repository Structure

```
NLP_Interview/
├── README.md                    # Main documentation
├── CLAUDE.md                    # This file
├── NLP/                         # 27 problem directories
│   ├── Attention_Mechanisms/
│   ├── TFIDF/
│   └── ...
├── docs/                        # Organized documentation
│   ├── anki/                    # Anki guides
│   ├── interview-guides/        # Interview preparation
│   └── study-plans/             # Study strategies
├── scripts/                     # Utility scripts
│   ├── problem_finder.py
│   ├── convert_to_anki*.py
│   └── clean_and_organize.py
└── anki_deck*/                  # Generated Anki decks
```

## Key Files for Context

- `README.md`: Complete overview with learning paths and company guides
- `docs/interview-guides/QUESTION_BANK_INDEX.md`: All problems indexed
- `docs/interview-guides/COMPANY_SPECIFIC_GUIDE.md`: Company focus areas
- `docs/interview-guides/SOLUTION_PATTERNS.md`: Reusable templates
- `scripts/problem_finder.py`: Interactive problem navigation