# Problem: Comprehensive Text Normalization Pipeline

Build a complete text normalization system:
1. `normalize_text(text: str, options: Dict[str, bool]) -> str`
2. `clean_html(html: str) -> str`
3. `expand_contractions(text: str) -> str`
4. `normalize_unicode(text: str) -> str`

Example:
Input: "I'll be there @ 3PM... Check https://example.com 😊"
Output: "I will be there at 3 PM. Check example.com"

Requirements:
- Handle URLs, emails, phone numbers
- Expand contractions and abbreviations
- Normalize whitespace and punctuation
- Support multiple languages

Follow-ups:
- Social media specific normalization (hashtags, mentions)
- Preserve important entities during cleaning
- Add spell correction
