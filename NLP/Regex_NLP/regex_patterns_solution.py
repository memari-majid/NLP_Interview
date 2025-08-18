import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import datetime


# Comprehensive regex pattern library
REGEX_PATTERNS = {
    # Email patterns
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    
    # Phone patterns (US format)
    'phone_us': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
    'phone_intl': r'\+\d{1,3}[-.\s]?\d{1,14}',
    
    # URL patterns
    'url': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
    
    # Date patterns
    'date_mdy': r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](\d{2,4})\b',
    'date_dmy': r'\b(0?[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])[/-](\d{2,4})\b',
    'date_ymd': r'\b(\d{4})[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])\b',
    'date_written': r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(0?[1-9]|[12][0-9]|3[01]),?\s+(\d{4})\b',
    
    # Time patterns
    'time_12': r'\b(0?[1-9]|1[0-2]):[0-5][0-9]\s*(AM|PM|am|pm)\b',
    'time_24': r'\b([01]?[0-9]|2[0-3]):[0-5][0-9]\b',
    
    # Money patterns
    'money_dollar': r'\$[\d,]*\.?\d+',
    'money_euro': r'€[\d,]*\.?\d+',
    'money_pound': r'£[\d,]*\.?\d+',
    'money_written': r'\b\d+\s*(dollars?|cents?|USD|EUR|GBP|pounds?)\b',
    
    # Percentage
    'percentage': r'\b\d+(?:\.\d+)?%',
    
    # Social Security Number
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    
    # Credit Card (simplified)
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    
    # IP Address
    'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
    
    # Hashtags and mentions
    'hashtag': r'#\w+',
    'mention': r'@\w+',
    
    # Abbreviations (common titles)
    'title_abbrev': r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.?\b',
}


def extract_entities_regex(text: str, 
                          custom_patterns: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
    """Extract various entities using regex patterns."""
    entities = defaultdict(list)
    
    # Use default patterns plus any custom ones
    patterns = REGEX_PATTERNS.copy()
    if custom_patterns:
        patterns.update(custom_patterns)
    
    # Extract emails
    emails = re.findall(patterns['email'], text, re.IGNORECASE)
    if emails:
        entities['EMAIL'] = list(set(emails))
    
    # Extract phone numbers
    phones = re.findall(patterns['phone_us'], text)
    phone_formatted = [f"({area})-{prefix}-{number}" for area, prefix, number in phones]
    intl_phones = re.findall(patterns['phone_intl'], text)
    all_phones = phone_formatted + intl_phones
    if all_phones:
        entities['PHONE'] = list(set(all_phones))
    
    # Extract URLs
    urls = re.findall(patterns['url'], text, re.IGNORECASE)
    if urls:
        entities['URL'] = list(set(urls))
    
    # Extract dates (multiple formats)
    dates = []
    for pattern_name in ['date_mdy', 'date_dmy', 'date_ymd', 'date_written']:
        matches = re.findall(patterns[pattern_name], text, re.IGNORECASE)
        dates.extend([' '.join(match) if isinstance(match, tuple) else match for match in matches])
    
    if dates:
        entities['DATE'] = list(set(dates))
    
    # Extract times
    times = []
    time_12 = re.findall(patterns['time_12'], text, re.IGNORECASE)
    time_24 = re.findall(patterns['time_24'], text)
    times.extend([f"{h}:{m} {ap}" for h, m, ap in time_12])
    times.extend(time_24)
    if times:
        entities['TIME'] = list(set(times))
    
    # Extract money amounts
    money = []
    for pattern_name in ['money_dollar', 'money_euro', 'money_pound', 'money_written']:
        matches = re.findall(patterns[pattern_name], text, re.IGNORECASE)
        money.extend(matches)
    
    if money:
        entities['MONEY'] = list(set(money))
    
    # Extract percentages
    percentages = re.findall(patterns['percentage'], text)
    if percentages:
        entities['PERCENTAGE'] = list(set(percentages))
    
    # Extract other entities
    for entity_type, pattern in [
        ('SSN', 'ssn'),
        ('CREDIT_CARD', 'credit_card'),
        ('IP_ADDRESS', 'ip_address'),
        ('HASHTAG', 'hashtag'),
        ('MENTION', 'mention')
    ]:
        matches = re.findall(patterns[pattern], text)
        if matches:
            entities[entity_type] = list(set(matches))
    
    return dict(entities)


def sentence_segmentation(text: str, preserve_abbreviations: bool = True) -> List[str]:
    """Segment text into sentences using regex, handling abbreviations."""
    
    # Common abbreviations that don't end sentences
    abbreviations = {
        'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr', 'vs', 'etc', 'eg', 'ie',
        'inc', 'ltd', 'corp', 'co', 'ave', 'st', 'blvd', 'rd', 'apt', 'dept',
        'fig', 'vol', 'no', 'pp', 'cf', 'al', 'ca', 'ny', 'tx', 'fl'
    }
    
    if preserve_abbreviations:
        # Replace abbreviations temporarily
        temp_text = text
        abbrev_placeholders = {}
        counter = 0
        
        for abbrev in abbreviations:
            pattern = fr'\b{re.escape(abbrev)}\.(?!\s*[A-Z])'
            matches = re.finditer(pattern, temp_text, re.IGNORECASE)
            
            for match in reversed(list(matches)):  # Reverse to maintain indices
                placeholder = f"__ABBREV_{counter}__"
                abbrev_placeholders[placeholder] = match.group()
                temp_text = temp_text[:match.start()] + placeholder + temp_text[match.end():]
                counter += 1
    else:
        temp_text = text
        abbrev_placeholders = {}
    
    # Handle ellipsis
    temp_text = re.sub(r'\.{2,}', '__ELLIPSIS__', temp_text)
    
    # Handle decimal numbers
    temp_text = re.sub(r'\b\d+\.\d+\b', lambda m: m.group().replace('.', '__DECIMAL__'), temp_text)
    
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+\s+', temp_text)
    
    # Clean up sentences
    cleaned_sentences = []
    for sentence in sentences:
        # Restore abbreviations
        for placeholder, original in abbrev_placeholders.items():
            sentence = sentence.replace(placeholder, original)
        
        # Restore ellipsis and decimals
        sentence = sentence.replace('__ELLIPSIS__', '...')
        sentence = sentence.replace('__DECIMAL__', '.')
        
        sentence = sentence.strip()
        if sentence:
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def pattern_based_ner(text: str, patterns: Dict[str, str]) -> List[Tuple[str, str, int, int]]:
    """Custom named entity recognition using regex patterns.
    
    Returns: List of (entity_text, entity_type, start_pos, end_pos)
    """
    entities = []
    
    for entity_type, pattern in patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            entity_text = match.group()
            start_pos = match.start()
            end_pos = match.end()
            
            entities.append((entity_text, entity_type, start_pos, end_pos))
    
    # Sort by start position
    entities.sort(key=lambda x: x[2])
    
    return entities


def clean_text_regex(text: str, rules: List[Tuple[str, str]]) -> str:
    """Apply multiple regex cleaning rules to text.
    
    Args:
        text: Input text to clean
        rules: List of (pattern, replacement) tuples
    """
    cleaned_text = text
    
    for pattern, replacement in rules:
        cleaned_text = re.sub(pattern, replacement, cleaned_text)
    
    return cleaned_text


def extract_structured_data(text: str) -> Dict[str, List[Dict]]:
    """Extract structured information like addresses, names, etc."""
    structured_data = defaultdict(list)
    
    # Address pattern (simplified US addresses)
    address_pattern = r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl)\.?(?:\s+(?:Apt|Apartment|Unit|Suite)\s*\w+)?'
    
    addresses = re.finditer(address_pattern, text, re.IGNORECASE)
    for match in addresses:
        structured_data['ADDRESS'].append({
            'text': match.group(),
            'start': match.start(),
            'end': match.end()
        })
    
    # Person name pattern (Title First Last)
    name_pattern = r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'
    
    names = re.finditer(name_pattern, text)
    for match in names:
        structured_data['PERSON_NAME'].append({
            'text': match.group(),
            'start': match.start(),
            'end': match.end()
        })
    
    # Product codes (letters and numbers)
    product_pattern = r'\b[A-Z]{2,3}-?\d{3,6}\b'
    
    products = re.finditer(product_pattern, text)
    for match in products:
        structured_data['PRODUCT_CODE'].append({
            'text': match.group(),
            'start': match.start(),
            'end': match.end()
        })
    
    return dict(structured_data)


def build_custom_tokenizer(patterns: Dict[str, str]) -> callable:
    """Build a custom tokenizer based on regex patterns."""
    
    def custom_tokenize(text: str) -> List[Tuple[str, str]]:
        """Tokenize text using custom patterns.
        
        Returns: List of (token, token_type) tuples
        """
        tokens = []
        remaining_text = text
        offset = 0
        
        while remaining_text:
            matched = False
            
            # Try each pattern
            for token_type, pattern in patterns.items():
                match = re.match(pattern, remaining_text)
                if match:
                    token = match.group()
                    tokens.append((token, token_type))
                    
                    # Update remaining text and offset
                    consumed = len(token)
                    remaining_text = remaining_text[consumed:]
                    offset += consumed
                    matched = True
                    break
            
            if not matched:
                # Default: consume one character as unknown
                if remaining_text:
                    tokens.append((remaining_text[0], 'UNKNOWN'))
                    remaining_text = remaining_text[1:]
                    offset += 1
        
        return tokens
    
    return custom_tokenize


def validate_extracted_data(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Validate extracted entities and filter out false positives."""
    validated = {}
    
    # Validate emails
    if 'EMAIL' in entities:
        valid_emails = []
        for email in entities['EMAIL']:
            # Additional validation beyond basic regex
            if '@' in email and '.' in email.split('@')[1]:
                parts = email.split('@')
                if len(parts) == 2 and parts[0] and parts[1]:
                    valid_emails.append(email)
        
        if valid_emails:
            validated['EMAIL'] = valid_emails
    
    # Validate phone numbers
    if 'PHONE' in entities:
        valid_phones = []
        for phone in entities['PHONE']:
            # Remove all non-digits
            digits_only = re.sub(r'\D', '', phone)
            # US phone should have 10 or 11 digits
            if len(digits_only) in [10, 11]:
                valid_phones.append(phone)
        
        if valid_phones:
            validated['PHONE'] = valid_phones
    
    # Validate dates
    if 'DATE' in entities:
        valid_dates = []
        for date_str in entities['DATE']:
            try:
                # Try to parse the date
                # This is a simplified validation
                if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', date_str):
                    valid_dates.append(date_str)
                elif re.match(r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}', date_str):
                    valid_dates.append(date_str)
            except:
                continue
        
        if valid_dates:
            validated['DATE'] = valid_dates
    
    # Copy other entities as-is
    for entity_type, entity_list in entities.items():
        if entity_type not in ['EMAIL', 'PHONE', 'DATE']:
            validated[entity_type] = entity_list
    
    return validated


def optimize_regex_performance(text: str, patterns: Dict[str, str]) -> Dict[str, List[str]]:
    """Optimized regex extraction for large texts."""
    entities = defaultdict(list)
    
    # Compile patterns once
    compiled_patterns = {name: re.compile(pattern, re.IGNORECASE) 
                        for name, pattern in patterns.items()}
    
    # Single pass through text
    for name, compiled_pattern in compiled_patterns.items():
        matches = compiled_pattern.findall(text)
        if matches:
            # Handle tuple matches (from groups)
            if matches and isinstance(matches[0], tuple):
                matches = ['-'.join(match) if isinstance(match, tuple) else match 
                          for match in matches]
            
            entities[name.upper()] = list(set(matches))
    
    return dict(entities)


# Common cleaning rules
COMMON_CLEANING_RULES = [
    # Remove extra whitespace
    (r'\s+', ' '),
    
    # Remove HTML tags
    (r'<[^>]+>', ''),
    
    # Remove URLs
    (r'https?://\S+', ''),
    
    # Remove email addresses
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ''),
    
    # Remove phone numbers
    (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ''),
    
    # Remove excessive punctuation
    (r'[!]{2,}', '!'),
    (r'[?]{2,}', '?'),
    (r'[.]{3,}', '...'),
    
    # Remove non-ASCII characters
    (r'[^\x00-\x7F]+', ' '),
]


if __name__ == "__main__":
    print("Regular Expression Patterns for NLP\n")
    
    # Example text with various entities
    sample_text = """
    Contact Dr. Smith at john.smith@email.com or call (555) 123-4567. 
    The meeting is scheduled for Jan 15, 2024 at 3:30 PM.
    
    Our office address is 123 Main Street, Apt 4B, New York, NY 10001.
    
    Please visit our website https://example.com for more information.
    The project budget is $25,000.50 with a 15% contingency.
    
    Product codes: AB-12345, XY-98765
    Credit card: 1234-5678-9012-3456
    IP Address: 192.168.1.1
    
    Follow us @company_account and use #NewProduct hashtag.
    """
    
    print("Sample Text:")
    print(sample_text)
    print("\n" + "="*60 + "\n")
    
    # Extract entities
    print("Extracted Entities:")
    entities = extract_entities_regex(sample_text)
    
    for entity_type, entity_list in entities.items():
        print(f"\n{entity_type}:")
        for entity in entity_list:
            print(f"  - {entity}")
    
    print("\n" + "="*60 + "\n")
    
    # Sentence segmentation
    print("Sentence Segmentation:")
    sentences = sentence_segmentation(sample_text)
    
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence.strip()}")
    
    print("\n" + "="*60 + "\n")
    
    # Custom pattern-based NER
    print("Pattern-based NER:")
    custom_patterns = {
        'PERSON_TITLE': r'\b(?:Dr|Mr|Mrs|Ms|Prof)\.?\s+[A-Z][a-z]+',
        'PRODUCT_CODE': r'\b[A-Z]{2}-\d{5}\b',
        'OFFICE_HOURS': r'\b\d{1,2}:\d{2}\s*(?:AM|PM)\b'
    }
    
    ner_results = pattern_based_ner(sample_text, custom_patterns)
    
    for entity_text, entity_type, start, end in ner_results:
        print(f"{entity_type}: '{entity_text}' ({start}-{end})")
    
    print("\n" + "="*60 + "\n")
    
    # Text cleaning
    print("Text Cleaning Example:")
    messy_text = "Check   out    our website!!! https://spam.com  Email: spam@example.com"
    print(f"Original: {messy_text}")
    
    cleaned = clean_text_regex(messy_text, COMMON_CLEANING_RULES)
    print(f"Cleaned:  {cleaned.strip()}")
    
    print("\n" + "="*60 + "\n")
    
    # Structured data extraction
    print("Structured Data Extraction:")
    structured = extract_structured_data(sample_text)
    
    for data_type, data_list in structured.items():
        print(f"\n{data_type}:")
        for item in data_list:
            print(f"  - {item['text']} (pos: {item['start']}-{item['end']})")
    
    print("\n" + "="*60 + "\n")
    
    # Custom tokenizer example
    print("Custom Tokenizer Example:")
    token_patterns = {
        'EMAIL': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
        'WORD': r'[A-Za-z]+',
        'NUMBER': r'\d+',
        'PUNCTUATION': r'[.,!?;:]',
        'WHITESPACE': r'\s+'
    }
    
    tokenizer = build_custom_tokenizer(token_patterns)
    tokens = tokenizer("Contact me at test@email.com, thanks!")
    
    print("Tokens:")
    for token, token_type in tokens:
        if token_type != 'WHITESPACE':  # Skip whitespace for display
            print(f"  '{token}' -> {token_type}")
    
    print("\n" + "="*60 + "\n")
    
    # Validation example
    print("Entity Validation:")
    raw_entities = {
        'EMAIL': ['john@email.com', 'invalid@', 'test@domain.co.uk'],
        'PHONE': ['555-123-4567', '123', '1-555-123-4567'],
        'DATE': ['01/15/2024', '13/40/2024', 'Jan 15, 2024']
    }
    
    print("Before validation:", raw_entities)
    validated = validate_extracted_data(raw_entities)
    print("After validation:", validated)