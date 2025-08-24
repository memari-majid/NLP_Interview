# APKG Workflow Guide

## 🎯 **Optimized Flashcard Creation Workflow**

This repository is optimized for creating high-quality flashcards that can be distributed as native APKG files for maximum user accessibility.

## 📁 **Repository Structure**

```
├── 📚 flashcards/                          # Source JSON files organized by collection
│   ├── NLP in Action/                       # ✅ Complete NLP collection (226 cards)
│   ├── nlp/                                 # General NLP topics (118 cards)
│   ├── ml/                                  # ML topics (35 cards) - Ready for expansion
│   └── [new_topic]/                         # Future collections
├── 📦 APKG_Output/                          # Generated APKG files ready for distribution
│   ├── Individual chapters (.apkg)          # One file per chapter/topic
│   └── Complete collections (.apkg)         # Master files combining multiple chapters
├── 🔄 convert_json_to_apkg.py              # Main conversion tool
├── 📝 Custom Instructions.md                # AI prompts for generating flashcard content
└── 📖 docs/                                # Documentation and guides
```

## 🚀 **Step-by-Step Workflow**

### **Step 1: Create Content (JSON Format)**

#### **Method A: AI Generation (Recommended)**
1. **Use Custom Instructions**: Copy prompts from `Custom Instructions.md`
2. **Provide source material**: Chapter, paper, documentation
3. **Generate with AI**: Use Claude, ChatGPT, etc.
4. **Get CrowdAnki JSON**: Complete with all required fields

#### **Method B: Manual Creation**
1. **Copy existing template**: Use any JSON file in `flashcards/` as template
2. **Follow the format**: Maintain all required CrowdAnki keys
3. **Structure answers**: Use 6-section format (Concept, Intuition, Mechanics, Trade-offs, Applications, Memory Hook)

### **Step 2: Organize JSON Files**

```
flashcards/[topic]/
├── 01_chapter_name/
│   └── 01_chapter_name.json     # Individual chapter
├── 02_chapter_name/
│   └── 02_chapter_name.json
└── ...
```

**Naming Convention:**
- **Directories**: `XX_topic_name` (e.g., `01_supervised_learning`)
- **JSON files**: Must match directory name exactly
- **Deck names in JSON**: Use hierarchical format `ML:[TOPIC]:XX Chapter Name`

### **Step 3: Convert to APKG**

```bash
python convert_json_to_apkg.py
```

**Options:**
1. **Individual files**: One APKG per chapter (recommended for hierarchical organization)
2. **Master file**: All chapters combined (good for complete topic collections)

### **Step 4: Distribute APKG Files**

- **Output location**: `APKG_Output/` directory
- **User instructions**: "Double-click to import into Anki"
- **No dependencies**: Works with vanilla Anki installation

## 🎨 **Content Quality Standards**

### **Question Format**
- **Interview-style**: "What is...", "How does...", "When would you..."
- **Single concept**: One idea per card
- **Realistic scenarios**: Questions you'd actually get asked

### **Answer Structure (Required)**
```html
<div class="concept"><strong>Concept:</strong> Core definition (1-2 lines)</div>
<div class="intuition"><strong>Intuition:</strong> Why it works (1-2 lines)</div>
<div class="mechanics"><strong>Mechanics:</strong> How it works (1-2 lines)</div>
<div class="tradeoffs"><strong>Trade-offs:</strong> Limitations (1-2 lines)</div>
<div class="applications"><strong>Applications:</strong> Real uses (1-2 lines)</div>
<div class="memory-hook"><strong>Memory Hook:</strong> Memorable phrase</div>
```

### **Difficulty Progression**
- **Easy**: Definitions, basic concepts
- **Medium**: Mechanics, applications, comparisons
- **Hard**: Mathematical details, edge cases, optimization

## 🔄 **Example: Creating ML Flashcards**

### **1. Set Up Structure**
```bash
mkdir -p "flashcards/ML Fundamentals"
mkdir -p "flashcards/ML Fundamentals/01_supervised_learning"
mkdir -p "flashcards/ML Fundamentals/02_unsupervised_learning"
# ... etc
```

### **2. Generate Content**
- Use `Custom Instructions.md` with ML textbook chapter
- Generate JSON with hierarchical naming: `ML:ML_Fundamentals:01 Supervised Learning`

### **3. Convert and Distribute**
```bash
python convert_json_to_apkg.py
# Results in APKG_Output/01-supervised-learning.apkg, etc.
```

### **4. User Experience**
- Download APKG files
- Double-click to import
- Study with perfect ML:ML_Fundamentals:XX hierarchy in Anki

## 🎯 **Best Practices**

### **For Content Creation**
- **Start with quality source**: Use authoritative textbooks, papers, courses
- **One concept per card**: Avoid information overload
- **Include examples**: Make abstract concepts concrete
- **Test understanding**: Review generated cards for accuracy

### **For Repository Organization**
- **Consistent naming**: Follow the established conventions
- **Logical grouping**: Group related topics together
- **Version control**: Commit JSON changes before conversion
- **Document sources**: Note what materials cards were based on

### **For Distribution**
- **Test before sharing**: Import APKG files to verify they work
- **Provide instructions**: Include simple "double-click to import" guidance
- **Consider packaging**: Zip related APKG files together
- **Update regularly**: Refresh content based on user feedback

## 🛠️ **Tools Reference**

### **Essential Files**
- **`convert_json_to_apkg.py`**: Main conversion script (requires `genanki`)
- **`Custom Instructions.md`**: AI prompts for content generation
- **Templates in `flashcards/`**: Copy structure from existing files

### **Dependencies**
```bash
pip install genanki  # For JSON to APKG conversion
```

### **Quality Checks**
- **JSON validation**: Use online JSON validators
- **Import testing**: Test APKG files in clean Anki installation
- **Mobile verification**: Check formatting on mobile devices

## 📊 **Success Metrics**

### **Content Quality**
- Cards can be answered in 10-30 seconds
- 80-90% success rate on mature cards
- Clear, memorable explanations
- Relevant to actual interview questions

### **User Experience**
- One-click APKG import works reliably
- Hierarchical organization appears correctly
- Cards display properly on all devices
- No technical setup required for end users

This workflow has been optimized based on successful creation and distribution of 226 NLP flashcards with excellent user feedback and adoption.
