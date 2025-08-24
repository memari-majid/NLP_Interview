# Flashcard Creation Guide

## Quick Start

This guide shows you how to create high-quality ML/NLP interview flashcards using our proven system.

## Method 1: Using Custom Instructions

### Step 1: Use the Custom Instructions

Copy the instructions from `Custom Instructions.md` and use them with any AI assistant (Claude, ChatGPT, etc.).

### Step 2: Provide Material

Give the AI assistant:
- Chapter text
- Research paper
- Lecture notes
- Technical documentation

### Step 3: Generate Flashcards

The AI will generate a complete JSON file in CrowdAnki format with:
- ✅ Interview-style questions
- ✅ Structured, memorable answers
- ✅ Proper difficulty levels
- ✅ Color-coded sections
- ✅ Mobile-optimized design

### Step 4: Save and Import

1. Save the JSON output as `[chapter_name].json`
2. Create directory with same name: `[chapter_name]/`
3. Put JSON file inside the directory
4. Import into Anki using CrowdAnki

## Answer Structure

Each flashcard follows this proven format:

### Front (Question)
```
What is the intuition behind gradient descent?
```

### Back (Answer)
```html
<div class="concept"><strong>Concept:</strong> Optimization algorithm that minimizes loss by moving opposite to gradient direction</div><br><br>
<div class="intuition"><strong>Intuition:</strong> Like rolling a ball downhill to find the bottom of a valley</div><br><br>
<div class="mechanics"><strong>Mechanics:</strong> θ = θ - α∇J(θ) where α is learning rate</div><br><br>
<div class="tradeoffs"><strong>Trade-offs:</strong> Fast convergence vs risk of overshooting; requires tuning learning rate</div><br><br>
<div class="applications"><strong>Applications:</strong> Training neural networks, logistic regression, SVM</div><br><br>
<div class="memory-hook"><strong>Memory Hook:</strong> Gradient = slope; Descent = going down; Always move downhill to minimize</div>
```

## Quality Guidelines

### Questions Should Be:
- **Interview-realistic**: "Explain the trade-offs of..."
- **Specific**: Focus on one concept per card
- **Actionable**: Can be answered in 30-60 seconds
- **Progressive**: Easy → Medium → Hard difficulty

### Answers Should Include:
1. **Concept**: Core definition (1 line)
2. **Intuition**: Why it works/what it feels like (1 line)
3. **Mechanics**: How it works/formula (1 line)
4. **Trade-offs**: Limitations and considerations (1 line)
5. **Applications**: Real-world uses (1 line)
6. **Memory Hook**: Memorable phrase or analogy (1 line)

## Difficulty Levels

### Easy (Definitions & Intuition)
- Basic definitions
- High-level concepts
- Intuitive explanations

### Medium (Mechanics & Trade-offs)
- How algorithms work
- When to use what
- Common trade-offs
- Implementation details

### Hard (Deep Reasoning & Math)
- Mathematical derivations
- Edge cases
- Comparisons between methods
- Optimization strategies

## Topic Coverage

### For Each Major Concept, Create:
- 1 Easy card (definition)
- 1-2 Medium cards (mechanics, applications)
- 1 Hard card (trade-offs, edge cases)

### Example: "Attention Mechanism"
1. **Easy**: "What is attention in NLP?" → Definition
2. **Medium**: "How does self-attention work?" → Mechanics
3. **Medium**: "What are common applications of attention?" → Applications
4. **Hard**: "Compare attention vs RNNs for long sequences" → Trade-offs

## File Organization

### Directory Structure
```
flashcards/
├── [book_name]/
│   ├── 01_chapter_name/
│   │   └── 01_chapter_name.json
│   ├── 02_chapter_name/
│   │   └── 02_chapter_name.json
└── [topic_area]/
    ├── Topic_Fundamentals/
    └── Topic_Advanced/
```

### Naming Conventions
- **Directories**: Use underscores and be descriptive
- **JSON Files**: Match directory name exactly
- **Deck Names**: Human-readable in JSON file

## Import Process

### Using CrowdAnki Add-on

1. **Install CrowdAnki**: Get from AnkiWeb add-ons
2. **Restart Anki**: Important for add-on activation
3. **File → Import**: Select directory (not JSON file)
4. **Choose Directory**: Select the folder containing JSON
5. **Import**: CrowdAnki handles the rest automatically

### Verification
- Check deck appears in Anki
- Review a few cards for formatting
- Test on mobile device
- Verify color-coded sections display

## Common Issues & Solutions

### Import Errors
- **Missing `__type__` fields**: Use our template exactly
- **JSON syntax errors**: Validate with JSONLint
- **Wrong directory structure**: JSON must be in named directory

### Display Issues
- **No colors**: CSS not loaded (check note model)
- **Poor mobile**: Use provided mobile-optimized CSS
- **Formatting broken**: Check HTML structure

### Content Quality
- **Too long**: Keep answers to 4-6 lines total
- **Too vague**: Be specific with examples
- **Missing context**: Include necessary background

## Best Practices

### Content Creation
- ✅ Use consistent question patterns
- ✅ Keep answers concise but complete
- ✅ Include memory hooks for difficult concepts
- ✅ Test understanding before publishing

### Deck Management
- ✅ Organize by learning objectives
- ✅ Use progressive difficulty
- ✅ Include tags for filtering
- ✅ Regular review and updates

### Study Strategy
- ✅ Start with Easy cards to build confidence
- ✅ Use spaced repetition consistently
- ✅ Focus on failed cards
- ✅ Apply knowledge through practice

## Advanced Tips

### For Large Topics
- Break into subtopics (one deck each)
- Create overview deck linking subtopics
- Use consistent tagging across decks

### For Math-Heavy Content
- Include formula cards with symbol definitions
- Create derivation cards for important proofs
- Add intuition cards for abstract concepts

### For Interview Prep
- Focus on "Why?" and "When?" questions
- Include comparison cards between methods
- Add implementation detail cards
- Practice explaining to others

## Example Workflow

1. **Read chapter/paper** thoroughly
2. **Identify key concepts** (5-10 per chapter)
3. **Use Custom Instructions** with AI assistant
4. **Review generated cards** for quality
5. **Create directory** and save JSON
6. **Import into Anki** and test
7. **Begin spaced repetition** immediately

## Quality Metrics

### Good Deck Signs
- 80-90% success rate on mature cards
- Clear, memorable answers
- Progressive difficulty
- Comprehensive topic coverage

### Red Flags
- <70% success rate (too hard)
- >95% success rate (too easy)
- Overly long answers
- Missing key concepts

This system has been tested across thousands of flashcards and consistently produces high-quality, memorable learning materials.
