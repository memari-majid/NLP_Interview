# LLM Flashcard Setup Guide

## 🎯 **Ready for ML:LLM:<chapters> Creation**

This directory contains the complete infrastructure for creating Large Language Model (LLM) flashcards with hierarchical organization in Anki.

## 📂 **Directory Structure**

```
flashcards/LLM/
├── _TEMPLATE.json              # Complete CrowdAnki template for LLM cards
├── 01_llm_foundations/         # What are LLMs, history, basic concepts
├── 02_transformer_architecture/# Architecture details, encoder-decoder
├── 03_attention_mechanisms/    # Self-attention, multi-head, positional encoding
├── 04_training_scaling/        # Pre-training, scaling laws, compute requirements
├── 05_fine_tuning_adaptation/  # SFT, RLHF, LoRA, adapters
├── 06_prompting_techniques/    # Few-shot, chain-of-thought, prompt engineering
├── 07_evaluation_benchmarks/   # BLEU, ROUGE, human eval, safety benchmarks
├── 08_deployment_inference/    # Model serving, optimization, quantization
├── 09_safety_alignment/        # Constitutional AI, red teaming, bias mitigation
└── 10_multimodal_llms/         # Vision-language, multimodal capabilities
```

## 🏗️ **Anki Hierarchy**

When imported, these flashcards will create:

```
📂 ML
  └── 📂 LLM  
      ├── 📚 01 LLM Foundations
      ├── 📚 02 Transformer Architecture
      ├── 📚 03 Attention Mechanisms  
      ├── 📚 04 Training & Scaling
      ├── 📚 05 Fine-tuning & Adaptation
      ├── 📚 06 Prompting Techniques
      ├── 📚 07 Evaluation & Benchmarks
      ├── 📚 08 Deployment & Inference
      ├── 📚 09 Safety & Alignment
      └── 📚 10 Multimodal LLMs
```

## 🚀 **Workflow**

### **1. Generate Content**
Use `Custom Instructions.md` with your AI assistant:
- Provide LLM chapter, paper, or documentation
- Specify: "Generate flashcards for ML:LLM:XX [Chapter Name]"
- Get complete CrowdAnki JSON output

### **2. Save & Convert**
```bash
# Save JSON file in appropriate directory
# Example: flashcards/LLM/01_llm_foundations/01_llm_foundations.json

# Convert to APKG
python convert_json_to_apkg.py

# Result: APKG_Output/01-llm-foundations.apkg
```

### **3. Import & Study**
- Double-click any `.apkg` file
- Anki imports with perfect ML:LLM hierarchy
- Study with spaced repetition

## 📋 **Content Guidelines**

### **Question Types**
- **Foundations**: "What is...", "How does..."
- **Technical**: "Compare X vs Y", "When to use..."  
- **Practical**: "How would you...", "What are the trade-offs..."

### **Answer Structure** (6 sections required)
1. **Concept**: Core definition
2. **Intuition**: Why it works
3. **Mechanics**: How it works  
4. **Trade-offs**: Limitations
5. **Applications**: Real uses
6. **Memory Hook**: Memorable phrase

### **Difficulty Distribution**
- **20% Easy**: Basic definitions and concepts
- **60% Medium**: Core mechanics and applications
- **20% Hard**: Advanced topics and edge cases

## 📚 **Recommended Sources**

### **Foundational Papers**
- "Attention Is All You Need" (Transformers)
- "Language Models are Few-Shot Learners" (GPT-3)
- "Training language models to follow instructions with human feedback" (InstructGPT)
- "Constitutional AI" (Claude/Anthropic)

### **Survey Papers**
- "A Survey of Large Language Models"
- "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods"
- "Multimodal Machine Learning: A Survey and Taxonomy"

### **Books & Courses**
- "Speech and Language Processing" by Jurafsky & Martin
- Stanford CS224N: Natural Language Processing with Deep Learning
- Hugging Face Transformers documentation

## 🎯 **Ready to Start**

Everything is set up for you to begin creating comprehensive LLM flashcards. Just pick a topic, use the updated Custom Instructions, and start generating!
