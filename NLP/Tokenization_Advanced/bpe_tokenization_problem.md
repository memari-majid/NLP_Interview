# Problem: Byte Pair Encoding (BPE) Tokenizer

**Time: 30 minutes**

Implement a simplified BPE tokenizer for subword segmentation.

```python
def build_bpe_vocab(texts: List[str], num_merges: int = 10) -> Dict[str, int]:
    """
    Build BPE vocabulary by iteratively merging most frequent pairs.
    
    Input: ["hello", "world", "hello"]
    Process:
        1. Start with characters: ['h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd']
        2. Find most frequent pair: 'l' + 'l' -> 'll'
        3. Merge and repeat
    
    Returns: Token to ID mapping
    """
    pass

def bpe_encode(text: str, vocab: Dict[str, int]) -> List[int]:
    """
    Encode text using BPE vocabulary.
    Apply merges greedily from longest to shortest.
    """
    pass
```

**Requirements:**
- Start with character-level vocabulary
- Iteratively merge most frequent adjacent pairs
- Build final vocabulary with token IDs
- Encode new text using learned merges

**Follow-up:** How does this handle out-of-vocabulary words better than word-level tokenization?
