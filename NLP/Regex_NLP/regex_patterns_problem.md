# Problem: Regular Expressions for NLP

Implement regex-based NLP functions:
1. `extract_entities_regex(text: str) -> Dict[str, List[str]]`
   - Extract emails, phones, URLs, dates, money amounts
2. `sentence_segmentation(text: str) -> List[str]`
   - Handle abbreviations, decimals, ellipsis
3. `pattern_based_ner(text: str, patterns: Dict[str, str]) -> List[Tuple[str, str, int, int]]`
   - Custom entity recognition with regex
4. `clean_text_regex(text: str, rules: List[Tuple[str, str]]) -> str`
   - Apply multiple cleaning rules

Example:
Text: "Contact Dr. Smith at john.smith@email.com or call (555) 123-4567. Meeting on Jan 15, 2024."
Entities: {
    "EMAIL": ["john.smith@email.com"],
    "PHONE": ["(555) 123-4567"],
    "DATE": ["Jan 15, 2024"]
}

Requirements:
- Handle edge cases (abbreviations, special formats)
- Support multiple date/time formats
- Extract structured information (prices, percentages)
- Build reusable pattern library

Follow-ups:
- Regex optimization for large texts
- Combining regex with ML models
- Multi-language pattern support
