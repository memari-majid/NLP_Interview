# Anki Integration Guide

This repository is designed to work seamlessly with Anki for spaced repetition memorization of NLP interview solutions.

## 🎯 What You Get

- **30+ Anki cards** automatically generated from problems/solutions
- **3 card types per problem**:
  - Implementation cards (problem → code solution)
  - Concept cards (topic → key points)
  - Complexity cards (algorithm → time/space analysis)
- **Organized by topic** with subdecks
- **Syntax highlighting** for code in Anki
- **Auto-sync** with GitHub updates

## 📦 Quick Setup (5 minutes)

### 1. Install Anki + CrowdAnki

1. Download [Anki Desktop](https://apps.ankiweb.net/) (free)
2. Open Anki → Tools → Add-ons → Get Add-ons
3. Enter code: `1788670778`
4. Restart Anki

### 2. Import the Deck

**Option A: From GitHub (Recommended)**
```
File → CrowdAnki: Import from GitHub
URL: https://github.com/[your-username]/ML_Coding
```

**Option B: From Local Files**
```bash
# First generate the deck
python convert_to_anki.py

# Then in Anki
File → CrowdAnki: Import from disk → Select 'anki_deck' folder
```

### 3. Start Studying!

- Anki will show 20 new cards/day by default
- Review cards based on your performance
- Cards get harder/easier based on your answers

## 🔄 Keeping Deck Updated

### Auto-Sync from GitHub

1. In Anki: `File → CrowdAnki: Import from GitHub`
2. It will merge updates while preserving your progress

### Manual Update

```bash
# Update problems/solutions in the repo
git pull origin master

# Regenerate Anki deck
python convert_to_anki.py

# Import in Anki (your progress is preserved)
File → CrowdAnki: Import from disk
```

## 📇 Card Types

### 1. Implementation Cards

**Front**: Problem statement  
**Back**: Complete solution with comments

Example:
```
Q: Implement TF-IDF from scratch
A: [Full implementation with step-by-step comments]
```

### 2. Concept Cards

**Front**: "What are the key concepts for [Algorithm]?"  
**Back**: Bullet points of key insights from comments

Example:
```
Q: What are the key concepts for Self-Attention?
A: • Scaled dot-product: QK^T/√d
   • Prevents gradient vanishing
   • O(n²) complexity
   • Causal masking for autoregressive
```

### 3. Complexity Cards

**Front**: "Time/Space complexity of [Algorithm]?"  
**Back**: Big-O notation with explanation

Example:
```
Q: Time/Space Complexity of TF-IDF?
A: Time: O(n*m) where n=docs, m=vocab
   Space: O(n*m) for TF-IDF matrix
```

## 🎨 Customization

### Modify Card Generation

Edit `convert_to_anki.py` to:
- Change card templates
- Add more card types
- Adjust code formatting
- Filter which problems to include

### Anki Settings

In Anki, you can:
- Change daily new card limit
- Adjust review intervals
- Modify card CSS styling
- Create filtered decks for cramming

## 📱 Mobile Study

1. Create free [AnkiWeb account](https://ankiweb.net/)
2. Sync desktop Anki with AnkiWeb
3. Use AnkiMobile (iOS) or AnkiDroid (Android)
4. Study on the go!

## 🧠 Optimal Study Strategy

### Phone Interview Prep (1 week)
- Set new cards to 10/day
- Focus on "implementation" tagged cards
- Use filtered deck for top 10 problems

### Deep Learning (1 month)
- Default 20 cards/day
- Study all card types
- Add your own cards for weak areas

### Last-Minute Review
- Create filtered deck: `tag:nlp_interview is:due`
- Cram mode for next interview
- Export to PDF for quick reference

## 🐛 Troubleshooting

**CrowdAnki not importing?**
- Ensure JSON is valid: `python -m json.tool anki_deck/NLP_Interview_Deck.json`
- Check Anki addon is installed and enabled

**Code not displaying correctly?**
- Update to latest Anki version
- Card CSS included in deck should handle formatting

**Want to contribute cards?**
- Add problems/solutions to NLP folders
- Run `convert_to_anki.py`
- Submit PR with updated deck

## 📊 Stats

Current deck contains:
- 26 topics
- 30+ cards
- 3 card types per problem
- Subdecks for organization

Happy studying! 🚀
