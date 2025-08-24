# Contributing to NLP & ML Interview Preparation Hub

We welcome contributions to improve our interview preparation materials! This guide will help you contribute effectively.

## 🎯 Ways to Contribute

### 1. **Flashcards**
- **Add new flashcards** to existing decks
- **Improve existing cards** for clarity and accuracy
- **Create new specialized decks** for niche topics
- **Fix formatting issues** in Anki JSON files

### 2. **Problems & Solutions**
- **Add new NLP problems** with comprehensive solutions
- **Improve existing solutions** for clarity and efficiency
- **Add alternative approaches** to existing problems
- **Update deprecated code** or libraries

### 3. **Documentation**
- **Improve explanations** in guides and READMEs
- **Add study strategies** and tips
- **Create tutorials** for specific topics
- **Fix typos** and formatting issues

### 4. **Question Banks**
- **Add new interview questions** from recent experiences
- **Categorize questions** by difficulty and topic
- **Provide detailed answers** with explanations
- **Update questions** for current industry trends

## 📋 Contribution Guidelines

### **Before You Start**
1. **Check existing issues** to avoid duplicates
2. **Create an issue** to discuss major changes
3. **Fork the repository** and create a feature branch
4. **Follow our naming conventions** (see below)

### **Quality Standards**

#### **For Flashcards:**
- **One concept per card** - Follow atomic learning principle
- **15-30 second review time** - Keep cards concise
- **Include symbols/definitions** - e.g., Q=query, K=key, V=value
- **Test on mobile** - Ensure readability on small screens
- **Verify Anki compatibility** - Test import before submitting

#### **For Code:**
- **Clean, readable code** with comments
- **Follow PEP 8** for Python code
- **Include time/space complexity** analysis
- **Test all solutions** before submitting
- **Add docstrings** for functions

#### **For Documentation:**
- **Clear, concise writing** at appropriate technical level
- **Use consistent formatting** (Markdown)
- **Include examples** where helpful
- **Maintain active voice** and professional tone

### **Naming Conventions**

#### **Flashcard Decks:**
- Format: `{Domain}_{Topic_Name}/`
- Examples: `NLP_Fundamentals/`, `ML_Deep_Learning/`
- Use underscores, no spaces or special characters

#### **Problem Files:**
- Format: `{topic_name}_problem.md` and `{topic_name}_solution.py`
- Use lowercase with underscores
- Be descriptive but concise

#### **Branch Names:**
- Format: `{type}/{description}`
- Types: `feature`, `bugfix`, `docs`, `refactor`
- Examples: `feature/add-transformer-cards`, `docs/update-readme`

## 🔄 Pull Request Process

### **1. Preparation**
```bash
# Fork and clone your fork
git clone https://github.com/YOUR_USERNAME/NLP_Interview.git
cd NLP_Interview

# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Test your changes
# - Import flashcards to Anki (if applicable)
# - Run any code solutions
# - Check documentation formatting
```

### **2. Commit Guidelines**
```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: add transformer attention mechanism flashcards

- Added 8 new cards covering self-attention
- Included mathematical formulas with symbol definitions
- Added practical examples and use cases
- Tested Anki import compatibility"

# Push to your fork
git push origin feature/your-feature-name
```

#### **Commit Message Format:**
- **Type**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`
- **Description**: Clear, concise summary (50 chars max)
- **Body**: Detailed explanation if needed
- **Examples**: 
  - `feat: add BERT explanation flashcards`
  - `fix: correct mathematical formula in attention cards`
  - `docs: update installation instructions`

### **3. Pull Request Requirements**
- **Clear title** describing the change
- **Detailed description** of what was added/changed
- **Testing notes** - how you verified the changes
- **Screenshots** if applicable (especially for flashcards)
- **Link to related issues** if applicable

### **4. Review Process**
1. **Automated checks** - ensure all files are properly formatted
2. **Content review** - verify accuracy and quality
3. **Testing** - check that Anki imports work (for flashcards)
4. **Feedback** - address any requested changes
5. **Merge** - once approved, your changes will be merged

## 🧪 Testing Your Contributions

### **For Flashcards:**
1. **Import to Anki** using CrowdAnki
2. **Review cards** for readability and timing
3. **Check mobile compatibility**
4. **Verify mathematical formulas** render correctly

### **For Problems:**
1. **Run all code solutions**
2. **Verify complexity analysis**
3. **Check edge cases**
4. **Ensure explanations are clear**

### **For Documentation:**
1. **Proofread for clarity** and accuracy
2. **Check all links** work correctly
3. **Verify formatting** in Markdown preview
4. **Test any instructions** step-by-step

## 📞 Getting Help

### **Questions?**
- **Create an issue** with the `question` label
- **Check existing discussions** in issues
- **Review documentation** in `docs/guides/`

### **Need Clarification?**
- **Comment on existing issues** for context
- **Tag maintainers** using `@username`
- **Be specific** about what you need help with

## 🏆 Recognition

Contributors will be:
- **Listed in README** contributors section
- **Tagged in release notes** for significant contributions
- **Given credit** in relevant documentation

## 📜 Code of Conduct

### **Our Standards**
- **Be respectful** and inclusive
- **Focus on constructive feedback**
- **Help others learn** and improve
- **Maintain professional communication**

### **Unacceptable Behavior**
- **Harassment or discrimination** of any kind
- **Trolling or inflammatory comments**
- **Publishing private information** without consent
- **Other conduct** inappropriate in a professional setting

---

**Thank you for contributing to help others succeed in their NLP and ML interviews! 🚀**
