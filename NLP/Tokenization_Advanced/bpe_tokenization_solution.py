from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import re

def get_word_frequencies(texts: List[str]) -> Dict[str, int]:
    """Get word frequencies with end-of-word marker."""
    word_freqs = Counter()
    
    for text in texts:
        words = text.lower().split()
        for word in words:
            # Add end-of-word marker
            word_with_marker = ' '.join(word) + ' </w>'
            word_freqs[word_with_marker] += 1
    
    return dict(word_freqs)

def get_pairs(word_freqs: Dict[str, int]) -> Counter:
    """Get all adjacent character pairs with their frequencies."""
    pairs = Counter()
    
    for word, freq in word_freqs.items():
        chars = word.split()
        for i in range(len(chars) - 1):
            pair = (chars[i], chars[i + 1])
            pairs[pair] += freq
    
    return pairs

def merge_vocab(pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
    """Merge most frequent pair in vocabulary."""
    new_word_freqs = {}
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word, freq in word_freqs.items():
        # Replace the pair with merged version
        new_word = pattern.sub(''.join(pair), word)
        new_word_freqs[new_word] = freq
    
    return new_word_freqs

def build_bpe_vocab(texts: List[str], num_merges: int = 10) -> Dict[str, int]:
    """Build BPE vocabulary through iterative merging."""
    if not texts:
        return {}
    
    # Initialize with character-level words
    word_freqs = get_word_frequencies(texts)
    
    # Start with character vocabulary
    vocab = set()
    for word in word_freqs:
        vocab.update(word.split())
    
    # Perform merges
    for i in range(num_merges):
        pairs = get_pairs(word_freqs)
        
        if not pairs:
            break
        
        # Get most frequent pair
        best_pair = pairs.most_common(1)[0][0]
        
        # Merge the pair
        word_freqs = merge_vocab(best_pair, word_freqs)
        
        # Add merged token to vocabulary
        vocab.add(''.join(best_pair))
        
        print(f"Merge {i+1}: {best_pair[0]} + {best_pair[1]} -> {''.join(best_pair)}")
    
    # Create token to ID mapping
    vocab_list = sorted(list(vocab))
    vocab_dict = {token: idx for idx, token in enumerate(vocab_list)}
    
    return vocab_dict

def bpe_encode(text: str, vocab: Dict[str, int]) -> List[int]:
    """Encode text using BPE vocabulary."""
    if not text:
        return []
    
    # Start with character-level splitting
    words = text.lower().split()
    encoded = []
    
    for word in words:
        # Convert word to character sequence with end marker
        word_chars = list(word) + ['</w>']
        
        # Greedily apply merges (simplified - just use available tokens)
        tokens = []
        i = 0
        
        while i < len(word_chars):
            # Try to find longest matching token
            found = False
            
            for length in range(min(len(word_chars) - i, 10), 0, -1):  # Max length 10
                candidate = ''.join(word_chars[i:i+length])
                
                if candidate in vocab:
                    tokens.append(vocab[candidate])
                    i += length
                    found = True
                    break
            
            if not found:
                # Fallback to unknown token (use ID 0)
                tokens.append(0)
                i += 1
        
        encoded.extend(tokens)
    
    return encoded

def bpe_decode(token_ids: List[int], vocab: Dict[str, int]) -> str:
    """Decode BPE tokens back to text."""
    # Create reverse vocabulary
    id_to_token = {idx: token for token, idx in vocab.items()}
    
    tokens = []
    for token_id in token_ids:
        if token_id in id_to_token:
            tokens.append(id_to_token[token_id])
    
    # Join tokens and handle end-of-word markers
    text = ''.join(tokens)
    text = text.replace('</w>', ' ')
    
    return text.strip()

def simulate_wordpiece(text: str, vocab: Dict[str, int]) -> List[str]:
    """Simulate WordPiece tokenization (greedy longest-match)."""
    words = text.lower().split()
    subwords = []
    
    for word in words:
        # Try to tokenize word using longest matching subwords
        i = 0
        word_subwords = []
        
        while i < len(word):
            # Find longest matching subword
            found = False
            
            for end in range(len(word), i, -1):
                subword = word[i:end]
                
                # Add ## prefix for continuation (except first subword)
                if i > 0:
                    subword = '##' + subword
                
                if subword in vocab or (i == 0 and subword in vocab):
                    word_subwords.append(subword)
                    i = end
                    found = True
                    break
            
            if not found:
                # Fallback: use [UNK] token
                word_subwords.append('[UNK]')
                i += 1
        
        subwords.extend(word_subwords)
    
    return subwords

# Test
if __name__ == "__main__":
    # Test BPE training
    print("BPE Tokenization Demo")
    print("=" * 40)
    
    # Training corpus
    training_texts = [
        "hello world",
        "hello there", 
        "world hello",
        "there hello"
    ]
    
    print("Training texts:", training_texts)
    
    # Build BPE vocabulary
    vocab = build_bpe_vocab(training_texts, num_merges=5)
    
    print(f"\nFinal vocabulary ({len(vocab)} tokens):")
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    for token, idx in sorted_vocab:
        print(f"  {idx}: '{token}'")
    
    print("\n" + "=" * 40)
    
    # Test encoding
    test_text = "hello world"
    encoded = bpe_encode(test_text, vocab)
    decoded = bpe_decode(encoded, vocab)
    
    print(f"Original: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    
    # Test OOV word
    oov_text = "goodbye world"
    encoded_oov = bpe_encode(oov_text, vocab)
    print(f"\nOOV test: '{oov_text}' -> {encoded_oov}")
    
    print("\n" + "=" * 40)
    
    # Demonstrate WordPiece-style tokenization
    print("WordPiece-style tokenization:")
    
    # Add some WordPiece-style tokens to vocab for demo
    wp_vocab = vocab.copy()
    wp_vocab.update({
        'hell': len(wp_vocab),
        '##o': len(wp_vocab) + 1,
        'wor': len(wp_vocab) + 2,
        '##ld': len(wp_vocab) + 3
    })
    
    wp_tokens = simulate_wordpiece("hello world", wp_vocab)
    print(f"WordPiece tokens: {wp_tokens}")
    
    print("\n" + "=" * 40)
    print("Key Differences:")
    print("• BPE: Merges most frequent character pairs")
    print("• WordPiece: Uses ## prefix for word continuations")
    print("• Both handle OOV better than word-level tokenization")
    print("• Subword tokens help with morphology and rare words")
