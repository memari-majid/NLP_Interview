# Hierarchical Deck Organization

## Overview

All Natural Language Processing in Action flashcard decks are organized using Anki's hierarchical naming system to create a folder-like structure for better organization and navigation.

## Hierarchy Structure

### Format: `ML:NLP:XX Chapter Name`

The hierarchical structure creates the following organization in Anki:

```
📂 ML (Machine Learning - Top Level)
  └── 📂 NLP (Natural Language Processing - Sublevel)
      ├── 📚 01 NLP Overview
      ├── 📚 02 Tokenization  
      ├── 📚 03 TF-IDF
      ├── 📚 04 Semantic Analysis
      ├── 📚 05 Neural Networks
      ├── 📚 06 Word Embeddings
      ├── 📚 07 CNNs
      ├── 📚 08 RNNs & LSTMs
      ├── 📚 09 Transformers
      ├── 📚 10 Large Language Models
      ├── 📚 11 Knowledge Graphs
      └── 📚 12 Dialog Engines
```

## Benefits of Hierarchical Organization

### 🎯 **Better Organization**
- **Logical grouping**: All NLP chapters under one parent folder
- **Scalable structure**: Easy to add more ML topics (Deep Learning, Computer Vision, etc.)
- **Clean interface**: Collapsed folders keep Anki deck list manageable

### 📱 **Enhanced Navigation**
- **Folder navigation**: Click to expand/collapse chapter groups
- **Quick access**: Find specific chapters faster
- **Mobile friendly**: Hierarchical structure works well on mobile Anki

### 🎓 **Study Benefits**
- **Progressive study**: Study entire ML:NLP collection or individual chapters
- **Topic focus**: Easy to focus on specific areas (e.g., just neural network chapters)
- **Visual clarity**: Clear separation between different subject areas

## Deck Names in Detail

| Full Name | Display in Anki | Content |
|-----------|----------------|---------|
| `ML:NLP:01 NLP Overview` | 01 NLP Overview | Natural Language Processing fundamentals |
| `ML:NLP:02 Tokenization` | 02 Tokenization | Text processing and tokenization |
| `ML:NLP:03 TF-IDF` | 03 TF-IDF | Term frequency-inverse document frequency |
| `ML:NLP:04 Semantic Analysis` | 04 Semantic Analysis | Finding meaning in word counts |
| `ML:NLP:05 Neural Networks` | 05 Neural Networks | Neural network foundations |
| `ML:NLP:06 Word Embeddings` | 06 Word Embeddings | Vector representations of words |
| `ML:NLP:07 CNNs` | 07 CNNs | Convolutional Neural Networks for text |
| `ML:NLP:08 RNNs & LSTMs` | 08 RNNs & LSTMs | Recurrent and LSTM networks |
| `ML:NLP:09 Transformers` | 09 Transformers | Attention mechanisms and transformers |
| `ML:NLP:10 Large Language Models` | 10 Large Language Models | Modern LLMs and applications |
| `ML:NLP:11 Knowledge Graphs` | 11 Knowledge Graphs | Information extraction and graphs |
| `ML:NLP:12 Dialog Engines` | 12 Dialog Engines | Conversational AI and chatbots |

## How to Use in Anki

### **Importing**
1. Import each chapter directory using CrowdAnki
2. Decks automatically appear in hierarchical structure
3. No manual organization needed

### **Studying Options**

#### **Study All NLP Chapters**
- Select the "NLP" parent folder
- Anki will include cards from all 12 chapters
- Great for comprehensive review

#### **Study Specific Chapters**
- Expand the NLP folder
- Select individual chapters (e.g., "05 Neural Networks")
- Focus on specific topics

#### **Study by Theme**
- Select multiple related chapters:
  - Chapters 1-4: Classical NLP foundations
  - Chapters 5-9: Modern neural approaches  
  - Chapters 10-12: Advanced applications

### **Mobile Experience**
- Tap folders to expand/collapse
- Clean, organized deck list
- Easy navigation even on small screens

## Future Expansion

This hierarchical structure allows for easy expansion:

### **Additional ML Topics**
```
📂 ML
  ├── 📂 NLP (Natural Language Processing)
  ├── 📂 CV (Computer Vision) - Future
  ├── 📂 RL (Reinforcement Learning) - Future
  └── 📂 DL (Deep Learning Fundamentals) - Future
```

### **NLP Subtopics**
```
📂 ML
  └── 📂 NLP
      ├── 📂 Fundamentals (Chapters 1-4)
      ├── 📂 Neural Networks (Chapters 5-9)
      └── 📂 Applications (Chapters 10-12)
```

## Technical Implementation

### **Anki Hierarchy Rules**
- Uses colon (`:`) separator to create hierarchy
- `ML:NLP:Chapter Name` creates: ML → NLP → Chapter Name
- Automatically creates parent folders if they don't exist
- Maintains numerical sorting within each level

### **Import Behavior**
- CrowdAnki preserves hierarchical names during import
- Existing hierarchy is maintained across imports
- Parent folders appear automatically in Anki interface

### **Compatibility**
- ✅ Works with Anki Desktop (Windows, Mac, Linux)  
- ✅ Works with AnkiMobile (iOS)
- ✅ Works with AnkiDroid (Android)
- ✅ Syncs properly across devices

## Best Practices

### **For Users**
- Keep hierarchy collapsed when not studying to reduce clutter
- Use parent folder selection for comprehensive reviews
- Bookmark frequently used individual chapters

### **For Content Creators**
- Maintain consistent naming pattern: `ML:NLP:XX Topic Name`
- Use descriptive but concise topic names
- Keep hierarchy depth reasonable (2-3 levels maximum)

This hierarchical organization transforms a flat list of 12 decks into a well-structured, navigable learning system that scales with your ML/NLP study needs.
