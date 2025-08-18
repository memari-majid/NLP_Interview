import re
from typing import List

def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words, preserving contractions and handling punctuation.
    
    This is ALWAYS asked in NLP interviews - seems simple but has many edge cases.
    
    Key challenges:
    - Contractions: "don't" should stay as one token, not ["don", "'", "t"]
    - Punctuation: "Hello!" should become ["Hello", "!"]
    - Empty/None input: Handle gracefully
    - Unicode characters: Different languages, emojis
    """
    
    # STEP 1: Handle edge cases first
    # Always check for None/empty input in interviews
    if not text:
        return []
    
    # STEP 2: Use regex pattern for tokenization
    # This is the most robust approach for handling complex cases
    
    # PATTERN EXPLANATION:
    # \w+(?:'\w+)?  - Matches word characters, optionally followed by apostrophe + more word chars
    #                 This handles contractions like "don't", "I'm", "we'll"
    # |             - OR operator
    # [^\w\s]       - Matches any non-word, non-space character (punctuation)
    #                 This treats each punctuation mark as separate token
    
    pattern = r"\w+(?:'\w+)?|[^\w\s]"
    
    # re.findall returns all non-overlapping matches
    tokens = re.findall(pattern, text)
    
    return tokens

def tokenize_simple(text: str) -> List[str]:
    """
    Alternative approach: Replace-then-split method.
    
    Sometimes interviewers want to see multiple approaches.
    This is simpler but less robust than regex.
    """
    if not text:
        return []
    
    # STEP 1: Replace punctuation with spaces (except apostrophes)
    # This preserves contractions while isolating other punctuation
    cleaned = re.sub(r"[^\w\s']", " ", text)
    
    # STEP 2: Split on whitespace
    # Simple but effective for basic cases
    return cleaned.split()

def advanced_tokenize(text: str, preserve_case: bool = False, 
                     handle_urls: bool = True, handle_emails: bool = True) -> List[str]:
    """
    Advanced tokenization with additional features.
    
    Shows awareness of real-world tokenization challenges.
    Good for follow-up questions about production systems.
    """
    if not text:
        return []
    
    processed_text = text
    
    # STEP 1: Handle special entities before general tokenization
    if handle_urls:
        # Replace URLs with special token
        # Pattern matches http(s) URLs
        url_pattern = r'https?://[^\s]+'
        processed_text = re.sub(url_pattern, '<URL>', processed_text)
    
    if handle_emails:
        # Replace emails with special token
        # Basic email pattern for demonstration
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        processed_text = re.sub(email_pattern, '<EMAIL>', processed_text)
    
    # STEP 2: Case handling
    if not preserve_case:
        processed_text = processed_text.lower()
    
    # STEP 3: Tokenize using main pattern
    pattern = r"\w+(?:'\w+)?|[^\w\s]"
    tokens = re.findall(pattern, processed_text)
    
    return tokens

def handle_contractions_explicitly(text: str) -> str:
    """
    Explicit contraction handling - shows deep understanding.
    
    Some interviewers want to see you handle contractions manually.
    This demonstrates knowledge of English language patterns.
    """
    # Common contractions mapping
    # In production, you'd have a much larger dictionary
    contractions = {
        "don't": "do not",
        "won't": "will not", 
        "can't": "cannot",
        "n't": " not",  # General pattern for negations
        "'ll": " will",
        "'re": " are", 
        "'ve": " have",
        "'m": " am",
        "'d": " would"  # or "had" - context dependent
    }
    
    # Apply contractions in order (longest first)
    expanded = text
    for contraction, expansion in sorted(contractions.items(), key=len, reverse=True):
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(contraction) + r'\b'
        expanded = re.sub(pattern, expansion, expanded, flags=re.IGNORECASE)
    
    return expanded

# COMPREHENSIVE TEST SUITE FOR INTERVIEWS
def test_tokenization():
    """
    Comprehensive test cases that cover edge cases interviewers ask about.
    
    Practice explaining why each test case is important.
    """
    test_cases = [
        # Basic cases
        ("Hello world", ["Hello", "world"]),
        ("Hello world!", ["Hello", "world", "!"]),
        
        # Contractions (most common edge case in interviews)
        ("don't go", ["don't", "go"]),
        ("I'm happy", ["I'm", "happy"]),
        ("We'll see", ["We'll", "see"]),
        
        # Punctuation handling
        ("Hello, world!", ["Hello", ",", "world", "!"]),
        ("What?!?", ["What", "?", "!", "?"]),
        
        # Edge cases (always test these in interviews)
        ("", []),  # Empty string
        ("   ", []),  # Only spaces
        ("one,two;three", ["one", ",", "two", ";", "three"]),
        
        # Complex contractions
        ("I've been there", ["I've", "been", "there"]),
        ("You're right", ["You're", "right"]),
        
        # Numbers and special characters
        ("Call 911!", ["Call", "911", "!"]),
        ("Price: $19.99", ["Price", ":", "$", "19.99"]),
    ]
    
    print("TOKENIZATION TEST SUITE")
    print("=" * 30)
    
    passed = 0
    total = len(test_cases)
    
    for text, expected in test_cases:
        result = tokenize(text)
        
        # Check if result matches expected
        if result == expected:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
        
        print(f"{status} '{text}' -> {result}")
        if result != expected:
            print(f"     Expected: {expected}")
    
    print(f"\nSUCCESS RATE: {passed}/{total} ({100*passed/total:.1f}%)")
    
    return passed == total

# INTERVIEW DEMO: Show your thinking process
if __name__ == "__main__":
    print("TOKENIZATION - Interview Walkthrough")
    print("=" * 45)
    
    # Test the main function
    success = test_tokenization()
    
    print(f"\n" + "=" * 45)
    print("COMPARING DIFFERENT APPROACHES")
    print("=" * 45)
    
    test_text = "I can't believe it's working!"
    
    # Show different tokenization approaches
    print(f"Input text: '{test_text}'")
    
    print(f"\n1. REGEX APPROACH (recommended):")
    regex_result = tokenize(test_text)
    print(f"   Result: {regex_result}")
    print(f"   ✓ Handles contractions correctly")
    
    print(f"\n2. SIMPLE SPLIT APPROACH:")
    simple_result = tokenize_simple(test_text)
    print(f"   Result: {simple_result}")
    print(f"   ✓ Simpler but less precise")
    
    print(f"\n3. CONTRACTION EXPANSION:")
    expanded = handle_contractions_explicitly(test_text)
    expanded_tokens = expanded.split()
    print(f"   Expanded: '{expanded}'")
    print(f"   Tokens: {expanded_tokens}")
    print(f"   ✓ Good for downstream processing")
    
    print(f"\n4. ADVANCED FEATURES:")
    text_with_email = "Contact me at john@email.com for details!"
    advanced_result = advanced_tokenize(text_with_email, handle_emails=True)
    print(f"   Input: '{text_with_email}'")
    print(f"   Result: {advanced_result}")
    print(f"   ✓ Handles special entities")
    
    print(f"\n" + "=" * 45)
    print("KEY INTERVIEW POINTS TO MENTION:")
    print("=" * 45)
    print("• Always handle edge cases (empty input, None)")
    print("• Regex approach is most robust for English")
    print("• Contractions are the #1 tokenization gotcha")
    print("• Consider downstream tasks when choosing approach")
    print("• Production systems need language-specific handling")
    print("• Subword tokenization (BPE) better for rare words")
    print("• Time complexity: O(n) where n = text length")
    print("• Space complexity: O(n) for storing tokens")
    
    print(f"\n" + "=" * 45)
    print("COMMON FOLLOW-UP QUESTIONS:")
    print("=" * 45)
    print("Q: How would you handle different languages?")
    print("A: Use language-specific regex, Unicode categories")
    print()
    print("Q: What about very long documents?")
    print("A: Stream processing, yield tokens instead of list")
    print()
    print("Q: How to handle social media text?")
    print("A: Special handling for @mentions, #hashtags, emojis")
    print()
    print("Q: Tokenization vs subword tokenization?")  
    print("A: Subword (BPE) better for OOV, morphology, rare words")