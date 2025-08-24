# Deck Naming Standards

## Overview

All flashcard decks in the Natural Language Processing in Action collection follow a standardized naming convention for optimal display and sorting in Anki.

## Naming Convention

### Format: `XX: Short Topic Name`

Where:
- `XX` = Zero-padded chapter number (01, 02, 03, etc.)
- `:` = Separator for clean visual distinction
- `Short Topic Name` = Concise, memorable topic description

### Benefits of This Format

1. **Perfect Sorting**: Zero-padded numbers ensure proper numerical order (01, 02, 03... not 1, 10, 11, 2...)
2. **Visual Clarity**: Colon separator makes the hierarchy clear
3. **Mobile Friendly**: Short names fit well on small screens
4. **Memorable**: Concise but descriptive topic names
5. **Consistent**: Same format across all 12 decks

## Current Deck Names

| Chapter | Deck Name | Topic |
|---------|-----------|--------|
| 01 | `01: NLP Overview` | Natural Language Processing fundamentals |
| 02 | `02: Tokenization` | Text processing and tokenization |
| 03 | `03: TF-IDF` | Term frequency-inverse document frequency |
| 04 | `04: Semantic Analysis` | Finding meaning in word counts |
| 05 | `05: Neural Networks` | Neural network foundations |
| 06 | `06: Word Embeddings` | Vector representations of words |
| 07 | `07: CNNs` | Convolutional Neural Networks for text |
| 08 | `08: RNNs & LSTMs` | Recurrent and Long Short-Term Memory networks |
| 09 | `09: Transformers` | Attention mechanisms and transformer models |
| 10 | `10: Large Language Models` | Modern LLMs and their applications |
| 11 | `11: Knowledge Graphs` | Information extraction and graph structures |
| 12 | `12: Dialog Engines` | Conversational AI and chatbot systems |

## Display in Anki

With this naming convention, decks will appear in Anki as:

```
01: NLP Overview                    (20 cards)
02: Tokenization                    (20 cards) 
03: TF-IDF                         (16 cards)
04: Semantic Analysis              (20 cards)
05: Neural Networks                (20 cards)
06: Word Embeddings                (15 cards)
07: CNNs                           (20 cards)
08: RNNs & LSTMs                   (20 cards)
09: Transformers                   (10 cards)
10: Large Language Models          (3 cards)
11: Knowledge Graphs               (X cards)
12: Dialog Engines                 (X cards)
```

## Key Features

### Sorting Behavior
- ✅ **Numerical order**: 01, 02, 03... (not 1, 10, 11, 2...)
- ✅ **Alphabetical backup**: If numbers were missing, would still sort correctly
- ✅ **Cross-platform**: Works consistently on desktop and mobile

### Visual Design
- ✅ **Scannable**: Easy to find specific topics quickly
- ✅ **Compact**: Fits well in Anki's deck list interface
- ✅ **Professional**: Clean, consistent appearance

### User Experience
- ✅ **Intuitive**: Numbers clearly indicate sequence
- ✅ **Memorable**: Topic names are recognizable
- ✅ **Study-friendly**: Easy to select specific areas for review

## Implementation

This naming standard is automatically applied to all decks in the collection. The format is enforced in:

1. **JSON `name` field**: Controls deck display name in Anki
2. **JSON `desc` field**: Provides longer description for deck info
3. **Directory structure**: Maintains consistent file organization

## For New Decks

When creating new decks for this collection:

1. **Follow the pattern**: `XX: Short Topic Name`
2. **Use zero-padding**: 01, 02, not 1, 2
3. **Keep names under 25 characters** when possible
4. **Use title case**: "Word Embeddings", not "word embeddings"
5. **Be descriptive but concise**: "CNNs" not "Convolutional Neural Networks for Text Processing"

## Migration Notes

All existing decks have been updated from their original long names:

- ❌ Old: "1 Machines that read and write: A natural language processing overview"
- ✅ New: "01: NLP Overview"

This change improves:
- Mobile display (fits on phone screens)
- Scan efficiency (faster to find specific topics)
- Professional appearance (clean, consistent formatting)
- Sorting reliability (numerical order guaranteed)

The standardized naming ensures optimal user experience across all Anki platforms and usage scenarios.
