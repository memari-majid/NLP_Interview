"""
Spell Checker - Anki-Optimized Solution
Each function is bite-sized for easy memorization.
Total cards generated: ~8 focused cards
"""

from typing import Dict, List, Tuple

# Card 1: Algorithm Overview
def spell_check_approach():
    """
    KEY: Use edit distance + frequency ranking
    STEPS: 1) Find close words 2) Rank by frequency
    INSIGHT: Most typos are 1-2 edits away
    """
    pass

# Card 2: Edit Distance Formula
def edit_distance_recursive(s1: str, s2: str) -> int:
    """
    FORMULA: ED(i,j) = min(
        ED(i-1,j) + 1,    # deletion
        ED(i,j-1) + 1,    # insertion  
        ED(i-1,j-1) + 0/1 # substitution
    )
    """
    if not s1: return len(s2)
    if not s2: return len(s1)
    
    if s1[0] == s2[0]:
        return edit_distance_recursive(s1[1:], s2[1:])
    
    return 1 + min(
        edit_distance_recursive(s1[1:], s2),    # delete
        edit_distance_recursive(s1, s2[1:]),    # insert
        edit_distance_recursive(s1[1:], s2[1:]) # replace
    )

# Card 3: Optimized Edit Distance
def edit_distance_dp(s1: str, s2: str) -> int:
    """
    COMPLEXITY: O(m*n) time, O(m*n) space
    OPTIMIZE: Can reduce to O(min(m,n)) space
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    
    # Base cases
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    
    # Fill matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # delete
                    dp[i][j-1],    # insert
                    dp[i-1][j-1]   # replace
                )
    
    return dp[m][n]

# Card 4: Find Candidates
def find_candidates(word: str, dictionary: Dict[str, int], 
                   max_distance: int = 2) -> List[Tuple[str, int]]:
    """
    STRATEGY: Only check words within length ±2
    EDGE: Empty dictionary returns empty list
    """
    if not dictionary:
        return []
    
    candidates = []
    word_len = len(word)
    
    for dict_word in dictionary:
        # Pruning: skip if length difference > max_distance
        if abs(len(dict_word) - word_len) > max_distance:
            continue
            
        distance = edit_distance_dp(word, dict_word)
        if distance <= max_distance:
            candidates.append((dict_word, distance))
    
    return candidates

# Card 5: Rank by Frequency
def rank_by_frequency(candidates: List[Tuple[str, int]], 
                     dictionary: Dict[str, int]) -> List[str]:
    """
    FORMULA: score = frequency / (distance + 1)
    KEY: Higher frequency, lower distance = better
    """
    scored = []
    
    for word, distance in candidates:
        freq = dictionary.get(word, 1)
        score = freq / (distance + 1)
        scored.append((word, score))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return [word for word, _ in scored[:3]]

# Card 6: Main Function
def spell_checker(word: str, dictionary: Dict[str, int]) -> List[str]:
    """
    INTERVIEW: Mention trade-offs:
    - Edit distance vs phonetic matching
    - Memory vs speed (DP vs recursive)
    - Pruning strategies for large dictionaries
    """
    # EDGE: Already correct word
    if word in dictionary:
        return [word]
    
    # EDGE: Empty dictionary
    if not dictionary:
        return []
    
    # Find candidates within edit distance 2
    candidates = find_candidates(word, dictionary, max_distance=2)
    
    # EDGE: No candidates found
    if not candidates:
        return []
    
    # Rank by frequency and return top 3
    return rank_by_frequency(candidates, dictionary)

# Card 7: Complexity Analysis
"""
COMPLEXITY ANALYSIS:
- find_candidates: O(n*m²) where n=dict size, m=word length
- rank_by_frequency: O(k log k) where k=candidates
- Overall: O(n*m²) dominated by edit distance

OPTIMIZATION IDEAS:
- Use BK-tree for faster candidate search
- Precompute common misspellings
- Use Levenshtein automaton for O(n) search
"""

# Card 8: Example Usage
def example_usage():
    """
    EXAMPLE:
    dictionary = {"hello": 100, "help": 80, "hell": 20}
    spell_checker("helo", dictionary) → ["hello", "help", "hell"]
    
    TEST EDGE CASES:
    - Empty string
    - Word already in dictionary  
    - No close matches
    - Single character words
    """
    pass
