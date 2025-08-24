# Repository Maintenance Guide

## Overview

This guide documents how to maintain, update, and expand the NLP/ML Interview Preparation Hub repository for future development.

## Repository Structure

### Current Organization
```
📁 NLP_Interview/
├── 📚 flashcards/                      # Main flashcard collections
│   ├── Natural Language Processing in Action/  # Book-based series
│   ├── nlp/                            # General NLP topics
│   └── ml/                             # Machine Learning topics
├── 📖 docs/                            # Documentation
│   ├── guides/                         # Study and reference guides
│   └── study-plans/                    # Learning pathways
├── 🛠️ utilities/                       # Helper scripts
├── 📊 assets/                          # Supporting data
└── 📝 Custom Instructions.md           # AI generation prompts
```

### File Standards
- **JSON files**: Must follow CrowdAnki format exactly
- **Documentation**: Markdown with clear headers and examples
- **Naming**: Consistent underscore convention
- **Encoding**: UTF-8 for all text files

## Quality Standards

### Flashcard Quality Requirements
- ✅ **CrowdAnki compatible** - Imports without errors
- ✅ **Mobile optimized** - Readable on small screens
- ✅ **Structured answers** - All 6 sections present
- ✅ **Appropriate difficulty** - Easy/Medium/Hard balanced
- ✅ **Interview relevant** - Realistic technical questions

### JSON Validation Checklist
- [ ] `"__type__": "Deck"` at root level
- [ ] `"children": []` present
- [ ] All required keys included
- [ ] Valid JSON syntax (JSONLint verified)
- [ ] Unique IDs throughout
- [ ] Proper HTML formatting in answers

## Adding New Content

### Method 1: AI-Generated Content (Recommended)

1. **Prepare source material** (chapter, paper, documentation)
2. **Use Custom Instructions.md** with AI assistant
3. **Validate output** against quality checklist
4. **Create directory** matching JSON filename
5. **Test import** in Anki before committing

### Method 2: Manual Creation

1. **Copy existing template** from working deck
2. **Follow CrowdAnki Format Guide** exactly
3. **Use structured answer format** for all cards
4. **Validate JSON** with online tools
5. **Test import** thoroughly

### Content Guidelines

#### Questions Should:
- Be interview-realistic ("Explain the trade-offs of...")
- Focus on single concepts
- Use progressive difficulty
- Include context when needed

#### Answers Should:
- Follow 6-section structure religiously
- Keep each section to 1-2 lines maximum
- Include specific examples
- End with memorable hooks

## Maintenance Tasks

### Weekly
- [ ] Check for new issues and PRs
- [ ] Validate reported import problems
- [ ] Update documentation as needed
- [ ] Test random flashcard imports

### Monthly
- [ ] Review flashcard quality metrics
- [ ] Update study guides with new research
- [ ] Check for Anki/CrowdAnki compatibility updates
- [ ] Organize and tag new contributions

### Quarterly
- [ ] Full repository audit
- [ ] Update Custom Instructions based on feedback
- [ ] Refresh mobile optimization testing
- [ ] Plan new content areas

## Common Issues & Solutions

### Import Errors

#### `KeyError: 'children'`
- **Cause**: Missing children key in deck root
- **Fix**: Add `"children": []` to JSON
- **Prevention**: Use validated templates

#### `KeyError: '__type__'`
- **Cause**: Missing type annotations
- **Fix**: Add appropriate `__type__` fields throughout
- **Prevention**: Follow CrowdAnki Format Guide exactly

#### Invalid JSON Syntax
- **Cause**: Malformed JSON structure
- **Fix**: Validate with JSONLint and fix errors
- **Prevention**: Use JSON-aware editors

### Quality Issues

#### Cards Too Long
- **Issue**: Answers exceed mobile screen limits
- **Fix**: Break into multiple cards or condense
- **Standard**: Maximum 6 sections, 1-2 lines each

#### Poor Mobile Display
- **Issue**: Text too small or wide on phones
- **Fix**: Check CSS and formatting
- **Standard**: Test on actual mobile devices

#### Inconsistent Difficulty
- **Issue**: Difficulty levels don't match content complexity
- **Fix**: Review and adjust difficulty tags
- **Standard**: Easy (definitions), Medium (mechanics), Hard (trade-offs/math)

## Development Workflow

### For Contributors

1. **Fork repository** and create feature branch
2. **Follow quality standards** documented here
3. **Test thoroughly** before submitting PR
4. **Include documentation** for new features
5. **Reference issues** in commit messages

### For Maintainers

1. **Review PRs** against quality checklist
2. **Test imports** on clean Anki installation
3. **Validate documentation** updates
4. **Merge and tag** stable releases
5. **Update changelog** with notable changes

## Testing Protocol

### Before Adding New Content

1. **JSON Validation**:
   ```bash
   # Validate JSON syntax
   cat deck.json | python -m json.tool > /dev/null
   ```

2. **CrowdAnki Import Test**:
   - Fresh Anki installation
   - Import deck directory
   - Verify no errors
   - Check card display
   - Test mobile view

3. **Quality Review**:
   - All 6 answer sections present
   - Appropriate difficulty levels
   - Interview-relevant questions
   - Clear, concise explanations

### Automated Testing (Future)

Consider implementing:
- JSON schema validation
- Automated import testing
- Link checking for documentation
- Spell checking for content
- Mobile view screenshots

## Documentation Standards

### File Naming
- Use descriptive, uppercase names
- Include version date for time-sensitive guides
- Maintain consistent extension (.md for Markdown)

### Content Structure
- Start with clear overview
- Include examples for all instructions
- Provide troubleshooting sections
- End with next steps or references

### Link Management
- Use relative links within repository
- Check all links quarterly
- Include alt text for accessibility
- Document external dependencies

## Backup and Recovery

### Critical Files
- All JSON files in flashcards/ directories
- Custom Instructions.md (central to content generation)
- Documentation in docs/ (especially format guides)

### Backup Strategy
- Git version control (primary)
- Regular exports of working Anki decks
- Documentation snapshots before major changes
- Test imports on separate Anki profiles

## Future Enhancements

### Planned Improvements
- [ ] Automated JSON validation scripts
- [ ] Template generators for new topics
- [ ] Statistics dashboard for deck usage
- [ ] Integration with spaced repetition research
- [ ] Community contribution workflow

### Technical Debt
- [ ] Standardize all existing decks to current format
- [ ] Consolidate duplicate topics across collections
- [ ] Improve mobile CSS for edge cases
- [ ] Add comprehensive test suite

## Contact and Support

### For Contributors
- Use GitHub Issues for bugs and feature requests
- Tag maintainers for urgent import problems
- Provide full error messages and system info
- Include sample files when reporting format issues

### For Maintainers
- Monitor Issues daily during active development
- Respond to format questions within 24 hours
- Update documentation based on common questions
- Maintain quality standards consistently

This maintenance guide ensures the repository remains high-quality, accessible, and useful for ML/NLP interview preparation as it grows and evolves.
