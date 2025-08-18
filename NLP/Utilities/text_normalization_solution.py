import re
import unicodedata
from typing import Dict, List, Optional, Tuple
import html
from urllib.parse import urlparse


# Contraction mappings
CONTRACTIONS = {
    "i'll": "i will",
    "i've": "i have",
    "i'm": "i am",
    "i'd": "i would",
    "you'll": "you will",
    "you've": "you have",
    "you're": "you are",
    "you'd": "you would",
    "he'll": "he will",
    "he's": "he is",
    "he'd": "he would",
    "she'll": "she will",
    "she's": "she is",
    "she'd": "she would",
    "it'll": "it will",
    "it's": "it is",
    "it'd": "it would",
    "we'll": "we will",
    "we've": "we have",
    "we're": "we are",
    "we'd": "we would",
    "they'll": "they will",
    "they've": "they have",
    "they're": "they are",
    "they'd": "they would",
    "that'll": "that will",
    "that's": "that is",
    "that'd": "that would",
    "there's": "there is",
    "there'd": "there would",
    "who'll": "who will",
    "who's": "who is",
    "who'd": "who would",
    "what'll": "what will",
    "what's": "what is",
    "what'd": "what would",
    "where's": "where is",
    "where'd": "where would",
    "when's": "when is",
    "when'd": "when would",
    "why's": "why is",
    "why'd": "why would",
    "how's": "how is",
    "how'd": "how would",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "can't": "cannot",
    "hadn't": "had not",
    "haven't": "have not",
    "hasn't": "has not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "ain't": "is not",
    "let's": "let us",
    "'cause": "because",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "coulda": "could have",
    "shoulda": "should have",
    "woulda": "would have",
    "y'all": "you all"
}

# Common abbreviations
ABBREVIATIONS = {
    "@": "at",
    "&": "and",
    "w/": "with",
    "w/o": "without",
    "b/c": "because",
    "btw": "by the way",
    "fyi": "for your information",
    "asap": "as soon as possible",
    "eta": "estimated time of arrival",
    "diy": "do it yourself",
    "faq": "frequently asked questions",
    "rsvp": "please respond",
    "ps": "postscript",
    "eg": "for example",
    "ie": "that is",
    "etc": "et cetera",
    "vs": "versus",
    "mr": "mister",
    "mrs": "missus",
    "ms": "miss",
    "dr": "doctor",
    "prof": "professor",
    "sr": "senior",
    "jr": "junior"
}


def expand_contractions(text: str) -> str:
    """Expand contractions in text."""
    # Convert to lowercase for matching
    text_lower = text.lower()
    
    # Sort contractions by length (descending) to match longer ones first
    sorted_contractions = sorted(CONTRACTIONS.items(), key=lambda x: len(x[0]), reverse=True)
    
    for contraction, expansion in sorted_contractions:
        # Use word boundaries for accurate matching
        pattern = r'\b' + re.escape(contraction) + r'\b'
        text_lower = re.sub(pattern, expansion, text_lower, flags=re.IGNORECASE)
    
    # Preserve original capitalization pattern
    result = []
    for i, char in enumerate(text):
        if i < len(text_lower):
            if char.isupper():
                result.append(text_lower[i].upper())
            else:
                result.append(text_lower[i])
        else:
            result.append(char)
    
    return ''.join(result)


def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations."""
    for abbr, expansion in ABBREVIATIONS.items():
        # Handle period after abbreviation
        pattern1 = r'\b' + re.escape(abbr) + r'\.\b'
        pattern2 = r'\b' + re.escape(abbr) + r'\b'
        
        text = re.sub(pattern1, expansion, text, flags=re.IGNORECASE)
        text = re.sub(pattern2, expansion, text, flags=re.IGNORECASE)
    
    return text


def clean_html(html_text: str) -> str:
    """Remove HTML tags and decode entities."""
    # Remove script and style content
    html_text = re.sub(r'<script[^>]*>.*?</script>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
    html_text = re.sub(r'<style[^>]*>.*?</style>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML tags
    html_text = re.sub(r'<[^>]+>', ' ', html_text)
    
    # Decode HTML entities
    text = html.unescape(html_text)
    
    # Clean up whitespace
    text = ' '.join(text.split())
    
    return text


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters."""
    # Normalize to NFKD form
    text = unicodedata.normalize('NFKD', text)
    
    # Replace special quotes and dashes
    replacements = {
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '–': '-',
        '—': '-',
        '…': '...',
        '•': '*',
        '®': '(R)',
        '™': '(TM)',
        '©': '(C)',
        '°': ' degrees',
        '£': 'GBP',
        '€': 'EUR',
        '¥': 'JPY',
        '₹': 'INR'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove non-ASCII characters that can't be handled
    # text = ''.join(char if ord(char) < 128 else ' ' for char in text)
    
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Add space after punctuation if missing
    text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation in text."""
    # Replace multiple punctuation with single
    text = re.sub(r'\.{2,}', '.', text)  # Multiple dots to single
    text = re.sub(r'!{2,}', '!', text)   # Multiple exclamations
    text = re.sub(r'\?{2,}', '?', text)  # Multiple questions
    
    # Normalize ellipsis
    text = re.sub(r'\.{3,}', '...', text)
    
    # Fix common punctuation errors
    text = re.sub(r'\s*,\s*', ', ', text)  # Space after comma
    text = re.sub(r'\s*\.\s*', '. ', text)  # Space after period
    text = re.sub(r'\s*!\s*', '! ', text)   # Space after exclamation
    text = re.sub(r'\s*\?\s*', '? ', text)  # Space after question
    
    return text


def handle_urls(text: str, keep_domain: bool = True) -> str:
    """Handle URLs in text."""
    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    
    if keep_domain:
        def replace_url(match):
            url = match.group(0)
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.replace('www.', '')
                return domain
            except:
                return 'URL'
    else:
        def replace_url(match):
            return 'URL'
    
    text = re.sub(url_pattern, replace_url, text)
    return text


def handle_emails(text: str, anonymize: bool = True) -> str:
    """Handle email addresses in text."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    if anonymize:
        text = re.sub(email_pattern, 'EMAIL', text)
    else:
        # Keep domain only
        def replace_email(match):
            email = match.group(0)
            domain = email.split('@')[1]
            return f'user@{domain}'
        
        text = re.sub(email_pattern, replace_email, text)
    
    return text


def handle_phone_numbers(text: str) -> str:
    """Handle phone numbers in text."""
    # US phone numbers
    phone_patterns = [
        r'\b\+?1?\s*\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
        r'\b([0-9]{3})[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
    ]
    
    for pattern in phone_patterns:
        text = re.sub(pattern, 'PHONE', text)
    
    return text


def handle_numbers(text: str, spell_out: bool = False) -> str:
    """Handle numbers in text."""
    if spell_out:
        # Simple number to word conversion for small numbers
        num_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve'
        }
        
        for num, word in num_words.items():
            text = re.sub(r'\b' + num + r'\b', word, text)
    
    # Normalize number formats
    text = re.sub(r'\b(\d+),(\d+)\b', r'\1\2', text)  # Remove commas from numbers
    
    # Handle percentages
    text = re.sub(r'(\d+)\s*%', r'\1 percent', text)
    
    # Handle currency
    text = re.sub(r'\$\s*(\d+)', r'\1 dollars', text)
    
    return text


def handle_social_media(text: str, preserve_mentions: bool = True, preserve_hashtags: bool = True) -> str:
    """Handle social media specific elements."""
    if not preserve_mentions:
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
    else:
        # Normalize mentions
        text = re.sub(r'@(\w+)', r'USER_\1', text)
    
    if not preserve_hashtags:
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
    else:
        # Convert hashtags to normal words
        text = re.sub(r'#(\w+)', r'\1', text)
    
    # Handle retweet notation
    text = re.sub(r'\bRT\s+:', '', text)
    
    return text


def remove_emojis(text: str) -> str:
    """Remove emoji characters from text."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    
    return emoji_pattern.sub('', text)


def normalize_text(text: str, options: Optional[Dict[str, bool]] = None) -> str:
    """Main text normalization function with configurable options."""
    if options is None:
        options = {
            'lowercase': False,
            'expand_contractions': True,
            'expand_abbreviations': True,
            'handle_urls': True,
            'handle_emails': True,
            'handle_phones': True,
            'handle_numbers': True,
            'remove_emojis': True,
            'normalize_unicode': True,
            'normalize_whitespace': True,
            'normalize_punctuation': True,
            'handle_social_media': True
        }
    
    # Apply normalization steps in order
    if options.get('normalize_unicode', True):
        text = normalize_unicode(text)
    
    if options.get('expand_contractions', True):
        text = expand_contractions(text)
    
    if options.get('expand_abbreviations', True):
        text = expand_abbreviations(text)
    
    if options.get('handle_urls', True):
        text = handle_urls(text)
    
    if options.get('handle_emails', True):
        text = handle_emails(text)
    
    if options.get('handle_phones', True):
        text = handle_phone_numbers(text)
    
    if options.get('handle_numbers', True):
        text = handle_numbers(text)
    
    if options.get('remove_emojis', True):
        text = remove_emojis(text)
    
    if options.get('handle_social_media', True):
        text = handle_social_media(text)
    
    if options.get('normalize_punctuation', True):
        text = normalize_punctuation(text)
    
    if options.get('normalize_whitespace', True):
        text = normalize_whitespace(text)
    
    if options.get('lowercase', False):
        text = text.lower()
    
    return text


def batch_normalize(texts: List[str], options: Optional[Dict[str, bool]] = None) -> List[str]:
    """Normalize multiple texts with the same options."""
    return [normalize_text(text, options) for text in texts]


def create_custom_normalizer(preserve_entities: List[str]) -> callable:
    """Create a custom normalizer that preserves specific entities."""
    def custom_normalize(text: str) -> str:
        # Store entities and their positions
        preserved = []
        
        for entity in preserve_entities:
            pattern = re.escape(entity)
            for match in re.finditer(pattern, text, re.IGNORECASE):
                preserved.append((match.start(), match.end(), match.group()))
        
        # Sort by position (reverse order for replacement)
        preserved.sort(reverse=True)
        
        # Replace with placeholders
        for i, (start, end, entity) in enumerate(preserved):
            placeholder = f"__ENTITY_{i}__"
            text = text[:start] + placeholder + text[end:]
        
        # Normalize
        text = normalize_text(text)
        
        # Restore entities
        for i, (_, _, entity) in enumerate(preserved):
            placeholder = f"__ENTITY_{i}__"
            text = text.replace(placeholder, entity)
        
        return text
    
    return custom_normalize


def get_normalization_stats(original: str, normalized: str) -> Dict[str, int]:
    """Get statistics about the normalization process."""
    return {
        'original_length': len(original),
        'normalized_length': len(normalized),
        'characters_removed': len(original) - len(normalized),
        'original_words': len(original.split()),
        'normalized_words': len(normalized.split()),
        'reduction_percentage': round((1 - len(normalized) / len(original)) * 100, 2)
    }


if __name__ == "__main__":
    # Example 1: Basic normalization
    print("Example 1: Basic normalization")
    text1 = "I'll be there @ 3PM... Check https://example.com 😊"
    normalized1 = normalize_text(text1)
    print(f"Original:   '{text1}'")
    print(f"Normalized: '{normalized1}'")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Social media text
    print("Example 2: Social media normalization")
    text2 = "RT @user123: Can't believe it's already 2024! 🎉 #NewYear #2024Goals"
    normalized2 = normalize_text(text2)
    print(f"Original:   '{text2}'")
    print(f"Normalized: '{normalized2}'")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: HTML content
    print("Example 3: HTML cleaning")
    html_text = "<p>Hello <b>world</b>! Visit <a href='http://test.com'>our site</a>.</p>"
    cleaned = clean_html(html_text)
    print(f"HTML:    '{html_text}'")
    print(f"Cleaned: '{cleaned}'")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Contact information
    print("Example 4: Contact information")
    text4 = "Contact me at john.doe@email.com or call 555-123-4567"
    normalized4 = normalize_text(text4)
    print(f"Original:   '{text4}'")
    print(f"Normalized: '{normalized4}'")
    
    print("\n" + "="*50 + "\n")
    
    # Example 5: Custom options
    print("Example 5: Custom normalization options")
    text5 = "I'm EXCITED about the $100 discount!!!"
    
    # Keep case, don't expand contractions
    custom_options = {
        'lowercase': False,
        'expand_contractions': False,
        'expand_abbreviations': True,
        'handle_urls': True,
        'handle_emails': True,
        'handle_phones': True,
        'handle_numbers': True,
        'remove_emojis': True,
        'normalize_unicode': True,
        'normalize_whitespace': True,
        'normalize_punctuation': True
    }
    
    normalized5 = normalize_text(text5, custom_options)
    print(f"Original:   '{text5}'")
    print(f"Normalized: '{normalized5}'")
    
    print("\n" + "="*50 + "\n")
    
    # Example 6: Preserve specific entities
    print("Example 6: Preserve specific entities")
    text6 = "Apple Inc. announced iPhone 15 will cost $999"
    
    # Create normalizer that preserves company names
    preserve_normalizer = create_custom_normalizer(['Apple Inc.', 'iPhone 15'])
    normalized6 = preserve_normalizer(text6)
    print(f"Original:   '{text6}'")
    print(f"Normalized: '{normalized6}'")
    
    print("\n" + "="*50 + "\n")
    
    # Example 7: Normalization statistics
    print("Example 7: Normalization statistics")
    complex_text = """
    Hey @everyone! 😎 Can't believe it's already 2024... 
    Check out our AMAZING deals @ https://shop.example.com!!!
    Call 1-800-555-0123 or email support@example.com ASAP!
    """
    
    normalized_complex = normalize_text(complex_text)
    stats = get_normalization_stats(complex_text, normalized_complex)
    
    print("Original text:")
    print(complex_text.strip())
    print("\nNormalized text:")
    print(normalized_complex)
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
