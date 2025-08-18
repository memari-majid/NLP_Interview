import re
import math
from typing import Dict, List, Tuple
from collections import defaultdict


class VaderSentimentAnalyzer:
    """Rule-based sentiment analyzer inspired by VADER (Valence Aware Dictionary and sEntiment Reasoner)."""
    
    def __init__(self):
        # Basic sentiment lexicon with polarity scores (-4 to +4 range)
        self.lexicon = {
            # Positive words
            'good': 2.0, 'great': 3.0, 'excellent': 3.5, 'amazing': 3.5, 'awesome': 3.5,
            'fantastic': 3.5, 'wonderful': 3.0, 'perfect': 3.5, 'love': 3.0, 'like': 2.0,
            'enjoy': 2.5, 'happy': 2.5, 'pleased': 2.5, 'satisfied': 2.0, 'delighted': 3.0,
            'thrilled': 3.5, 'excited': 3.0, 'brilliant': 3.0, 'outstanding': 3.5,
            'superb': 3.5, 'magnificent': 3.5, 'marvelous': 3.0, 'splendid': 3.0,
            
            # Negative words
            'bad': -2.0, 'terrible': -3.0, 'awful': -3.5, 'horrible': -3.5, 'hate': -3.0,
            'dislike': -2.0, 'poor': -2.0, 'worst': -3.5, 'disgusting': -3.5, 'pathetic': -3.0,
            'disappointing': -2.5, 'sad': -2.5, 'angry': -2.5, 'furious': -3.5, 'annoyed': -2.0,
            'frustrated': -2.5, 'upset': -2.5, 'depressed': -3.0, 'miserable': -3.5,
            'dreadful': -3.0, 'appalling': -3.5, 'abysmal': -3.5,
            
            # Neutral/mild words
            'okay': 0.5, 'ok': 0.5, 'fine': 1.0, 'decent': 1.0, 'average': 0.0,
            'normal': 0.0, 'standard': 0.0, 'typical': 0.0, 'usual': 0.0,
        }
        
        # Intensifiers (boost sentiment)
        self.intensifiers = {
            'absolutely': 0.293, 'amazingly': 0.293, 'awfully': 0.293, 'completely': 0.293,
            'considerable': 0.293, 'decidedly': 0.293, 'deeply': 0.293, 'effing': 0.293,
            'enormously': 0.293, 'entirely': 0.293, 'especially': 0.293, 'exceptionally': 0.293,
            'extremely': 0.293, 'fabulously': 0.293, 'flipping': 0.293, 'flippin': 0.293,
            'fricking': 0.293, 'frickin': 0.293, 'frigging': 0.293, 'friggin': 0.293,
            'fully': 0.293, 'greatly': 0.293, 'hella': 0.293, 'highly': 0.293,
            'hugely': 0.293, 'incredibly': 0.293, 'intensely': 0.293, 'majorly': 0.293,
            'more': 0.293, 'most': 0.293, 'particularly': 0.293, 'purely': 0.293,
            'quite': 0.293, 'really': 0.293, 'remarkably': 0.293, 'so': 0.293,
            'substantially': 0.293, 'thoroughly': 0.293, 'totally': 0.293, 'tremendously': 0.293,
            'uber': 0.293, 'unbelievably': 0.293, 'unusually': 0.293, 'utterly': 0.293,
            'very': 0.293, 'wicked': 0.293
        }
        
        # Dampeners (reduce sentiment)
        self.dampeners = {
            'barely': -0.293, 'hardly': -0.293, 'just': -0.293, 'kind': -0.293,
            'kinda': -0.293, 'kindof': -0.293, 'kind_of': -0.293, 'less': -0.293,
            'little': -0.293, 'marginally': -0.293, 'occasionally': -0.293, 'partly': -0.293,
            'scarcely': -0.293, 'slightly': -0.293, 'somewhat': -0.293, 'sort': -0.293,
            'sorta': -0.293, 'sortof': -0.293, 'sort_of': -0.293
        }
        
        # Negation words
        self.negations = {
            'not', 'no', 'never', 'none', 'nothing', 'neither', 'nowhere', 'nobody',
            'cannot', 'cant', "can't", 'dont', "don't", 'isnt', "isn't", 'wasnt', "wasn't",
            'wont', "won't", 'wouldnt', "wouldn't", 'shouldnt', "shouldn't", 'couldnt', "couldn't"
        }
        
        # Punctuation emphasis multipliers
        self.punctuation_emphasis = {
            '!': 0.292,
            '?': 0.18,
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis."""
        # Convert to lowercase but preserve punctuation
        text = text.lower()
        
        # Handle contractions
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "isn't": "is not", "wasn't": "was not", "aren't": "are not",
            "weren't": "were not", "shouldn't": "should not", "wouldn't": "would not",
            "couldn't": "could not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization that preserves punctuation context
        words = re.findall(r'\w+|[!?]+', text)
        return words
    
    def _get_sentiment_score(self, word: str) -> float:
        """Get base sentiment score for a word."""
        return self.lexicon.get(word.lower(), 0.0)
    
    def _apply_intensifiers_and_dampeners(self, words: List[str], sentiment_scores: List[float]) -> List[float]:
        """Apply intensifiers and dampeners to sentiment scores."""
        enhanced_scores = sentiment_scores.copy()
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Look for intensifiers/dampeners in the previous 2 words
            for j in range(max(0, i-2), i):
                prev_word = words[j].lower()
                
                if prev_word in self.intensifiers and enhanced_scores[i] != 0:
                    # Boost sentiment
                    if enhanced_scores[i] > 0:
                        enhanced_scores[i] += self.intensifiers[prev_word]
                    else:
                        enhanced_scores[i] -= self.intensifiers[prev_word]
                
                elif prev_word in self.dampeners and enhanced_scores[i] != 0:
                    # Reduce sentiment
                    if enhanced_scores[i] > 0:
                        enhanced_scores[i] += self.dampeners[prev_word]  # Dampeners are negative
                    else:
                        enhanced_scores[i] -= self.dampeners[prev_word]
        
        return enhanced_scores
    
    def _apply_negation(self, words: List[str], sentiment_scores: List[float]) -> List[float]:
        """Apply negation rules to sentiment scores."""
        negated_scores = sentiment_scores.copy()
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Look for negations in the previous 3 words
            for j in range(max(0, i-3), i):
                prev_word = words[j].lower()
                
                if prev_word in self.negations:
                    # Flip and dampen the sentiment
                    if negated_scores[i] != 0:
                        negated_scores[i] *= -0.74  # Standard VADER negation factor
                    break  # Only apply the closest negation
        
        return negated_scores
    
    def _apply_punctuation_emphasis(self, text: str, base_sentiment: float) -> float:
        """Apply punctuation emphasis to overall sentiment."""
        emphasis_sum = 0
        
        for punct, emphasis in self.punctuation_emphasis.items():
            count = text.count(punct)
            if count > 0:
                # Cap the emphasis effect
                emphasis_sum += min(count, 4) * emphasis
        
        # Apply emphasis
        if base_sentiment > 0:
            return base_sentiment + emphasis_sum
        elif base_sentiment < 0:
            return base_sentiment - emphasis_sum
        else:
            return base_sentiment
    
    def _calculate_compound_score(self, sentiment_scores: List[float]) -> float:
        """Calculate the compound sentiment score."""
        sentiment_sum = sum(sentiment_scores)
        
        if sentiment_sum == 0:
            return 0.0
        
        # Normalize using VADER's alpha parameter
        alpha = 15
        compound = sentiment_sum / math.sqrt((sentiment_sum * sentiment_sum) + alpha)
        
        # Ensure compound score is in [-1, 1] range
        compound = max(-1, min(1, compound))
        
        return round(compound, 4)
    
    def _calculate_pos_neu_neg(self, sentiment_scores: List[float]) -> Tuple[float, float, float]:
        """Calculate positive, neutral, and negative proportions."""
        positive_sum = sum(score for score in sentiment_scores if score > 0)
        negative_sum = abs(sum(score for score in sentiment_scores if score < 0))
        neutral_count = sum(1 for score in sentiment_scores if score == 0)
        
        total_scores = len(sentiment_scores)
        
        if total_scores == 0:
            return 0.0, 1.0, 0.0
        
        # Calculate proportions
        if positive_sum + negative_sum == 0:
            pos = 0.0
            neg = 0.0
            neu = 1.0
        else:
            total_sentiment = positive_sum + negative_sum
            pos = positive_sum / total_sentiment if total_sentiment > 0 else 0.0
            neg = negative_sum / total_sentiment if total_sentiment > 0 else 0.0
            
            # Neutral proportion includes non-sentiment words
            neu = max(0.0, 1.0 - pos - neg)
        
        # Normalize to ensure they sum to 1
        total = pos + neu + neg
        if total > 0:
            pos /= total
            neu /= total
            neg /= total
        
        return round(pos, 3), round(neu, 3), round(neg, 3)
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text and return VADER-style scores."""
        if not text or not text.strip():
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Tokenize
        words = self._tokenize(processed_text)
        
        if not words:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        
        # Get base sentiment scores
        sentiment_scores = [self._get_sentiment_score(word) for word in words]
        
        # Apply intensifiers and dampeners
        sentiment_scores = self._apply_intensifiers_and_dampeners(words, sentiment_scores)
        
        # Apply negation
        sentiment_scores = self._apply_negation(words, sentiment_scores)
        
        # Calculate base compound score
        compound = self._calculate_compound_score(sentiment_scores)
        
        # Apply punctuation emphasis to compound score
        compound = self._apply_punctuation_emphasis(text, compound)
        compound = max(-1, min(1, compound))  # Ensure bounds
        
        # Calculate pos, neu, neg proportions
        pos, neu, neg = self._calculate_pos_neu_neg(sentiment_scores)
        
        return {
            'compound': round(compound, 4),
            'pos': pos,
            'neu': neu,
            'neg': neg
        }
    
    def score_lexicon(self, words: List[str]) -> float:
        """Score a list of words based on the sentiment lexicon."""
        total_score = 0.0
        
        for word in words:
            score = self._get_sentiment_score(word)
            total_score += score
        
        return total_score
    
    def handle_intensifiers(self, words: List[str]) -> List[float]:
        """Apply intensifier rules and return boosted sentiment scores."""
        sentiment_scores = [self._get_sentiment_score(word) for word in words]
        return self._apply_intensifiers_and_dampeners(words, sentiment_scores)
    
    def add_to_lexicon(self, word: str, score: float):
        """Add a word and its sentiment score to the lexicon."""
        self.lexicon[word.lower()] = max(-4, min(4, score))  # Bound score
    
    def get_lexicon_stats(self) -> Dict[str, int]:
        """Get statistics about the sentiment lexicon."""
        positive_words = sum(1 for score in self.lexicon.values() if score > 0)
        negative_words = sum(1 for score in self.lexicon.values() if score < 0)
        neutral_words = sum(1 for score in self.lexicon.values() if score == 0)
        
        return {
            'total_words': len(self.lexicon),
            'positive_words': positive_words,
            'negative_words': negative_words,
            'neutral_words': neutral_words
        }


def compare_with_ml_approach(texts: List[str], true_labels: List[str]) -> Dict[str, float]:
    """Compare VADER with ML approach (simplified demo)."""
    vader = VaderSentimentAnalyzer()
    
    # VADER predictions
    vader_predictions = []
    for text in texts:
        scores = vader.analyze_sentiment(text)
        
        # Convert to categorical prediction
        if scores['compound'] >= 0.05:
            prediction = 'positive'
        elif scores['compound'] <= -0.05:
            prediction = 'negative'
        else:
            prediction = 'neutral'
        
        vader_predictions.append(prediction)
    
    # Calculate accuracy
    correct = sum(1 for pred, true in zip(vader_predictions, true_labels) if pred == true)
    accuracy = correct / len(texts) if texts else 0
    
    return {
        'accuracy': accuracy,
        'predictions': vader_predictions
    }


def analyze_emoji_sentiment(text: str) -> Dict[str, float]:
    """Extend VADER to handle emoji sentiment (simplified)."""
    # Basic emoji sentiment mapping
    emoji_sentiment = {
        '😀': 2.0, '😁': 2.5, '😂': 3.0, '🤣': 3.0, '😃': 2.0, '😄': 2.5,
        '😅': 1.5, '😆': 2.5, '😇': 2.0, '🙂': 1.5, '🙃': 0.5, '😉': 1.5,
        '😊': 2.5, '😋': 2.0, '😌': 1.0, '😍': 3.0, '🥰': 3.0, '😘': 2.5,
        '😗': 2.0, '☺️': 2.0, '😚': 2.0, '😙': 2.0, '🥲': -0.5,
        
        '😢': -2.0, '😭': -3.0, '😤': -2.0, '😠': -2.5, '😡': -3.0,
        '🤬': -3.5, '😱': -2.0, '😨': -2.0, '😰': -2.5, '😥': -2.0,
        '😞': -2.0, '😔': -2.0, '😟': -2.0, '😕': -1.5, '🙁': -2.0,
        '☹️': -2.0, '😣': -2.0, '😖': -2.0, '😫': -2.5, '😩': -2.5,
        '🥺': -1.0, '😦': -1.5, '😧': -2.0, '😮': 0.0, '😯': 0.0
    }
    
    vader = VaderSentimentAnalyzer()
    
    # Add emoji scores to text analysis
    emoji_score = 0
    for emoji, score in emoji_sentiment.items():
        emoji_count = text.count(emoji)
        emoji_score += emoji_count * score
    
    # Get regular sentiment analysis
    base_scores = vader.analyze_sentiment(text)
    
    # Combine emoji sentiment with text sentiment
    combined_compound = base_scores['compound']
    if emoji_score != 0:
        # Normalize emoji score and add to compound
        emoji_normalized = emoji_score / (abs(emoji_score) + 15)  # Similar to VADER's alpha
        combined_compound = max(-1, min(1, combined_compound + emoji_normalized))
    
    return {
        'compound': round(combined_compound, 4),
        'pos': base_scores['pos'],
        'neu': base_scores['neu'],
        'neg': base_scores['neg'],
        'emoji_score': emoji_score
    }


if __name__ == "__main__":
    print("VADER-Style Rule-Based Sentiment Analysis\n")
    
    # Initialize analyzer
    vader = VaderSentimentAnalyzer()
    
    # Test sentences with different sentiment patterns
    test_sentences = [
        "This movie is absolutely fantastic!",
        "I hate this terrible product.",
        "It's okay, nothing special.",
        "Not bad at all, quite good actually.",
        "This is not good.",
        "I really don't like this very much.",
        "AMAZING!!! Best movie ever!!!",
        "It was kind of disappointing.",
        "Extremely poor quality.",
        "I'm so excited about this!",
    ]
    
    print("Sentiment Analysis Results:")
    print("-" * 80)
    print(f"{'Text':<40} {'Compound':<10} {'Pos':<6} {'Neu':<6} {'Neg':<6}")
    print("-" * 80)
    
    for sentence in test_sentences:
        scores = vader.analyze_sentiment(sentence)
        
        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'POSITIVE'
        elif scores['compound'] <= -0.05:
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'NEUTRAL'
        
        text_display = sentence[:37] + "..." if len(sentence) > 40 else sentence
        print(f"{text_display:<40} {scores['compound']:<10} {scores['pos']:<6} {scores['neu']:<6} {scores['neg']:<6} ({sentiment})")
    
    print("\n" + "="*60 + "\n")
    
    # Test component functions
    print("Component Analysis:")
    
    # Test intensifiers
    test_words = ["very", "good"]
    base_scores = [vader.score_lexicon([word]) for word in test_words]
    enhanced_scores = vader.handle_intensifiers(test_words)
    
    print(f"\nIntensifier Effect:")
    print(f"Words: {test_words}")
    print(f"Base scores: {base_scores}")
    print(f"Enhanced scores: {enhanced_scores}")
    
    # Test negation
    negation_examples = [
        "This is good",
        "This is not good",
        "This is not very good",
        "I don't really like this"
    ]
    
    print(f"\nNegation Examples:")
    for example in negation_examples:
        scores = vader.analyze_sentiment(example)
        print(f"'{example}': {scores['compound']}")
    
    print("\n" + "="*60 + "\n")
    
    # Lexicon statistics
    print("Lexicon Statistics:")
    stats = vader.get_lexicon_stats()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "="*60 + "\n")
    
    # Emoji sentiment analysis (if supported)
    print("Emoji Sentiment Analysis:")
    emoji_texts = [
        "I love this movie! 😍",
        "This is terrible 😢",
        "Great job! 🎉🎊",
        "I'm so angry 😡😡😡",
        "It's okay 😐"
    ]
    
    for text in emoji_texts:
        scores = analyze_emoji_sentiment(text)
        print(f"'{text}': compound={scores['compound']}, emoji_score={scores['emoji_score']}")
    
    print("\n" + "="*60 + "\n")
    
    # Domain adaptation example
    print("Domain Adaptation Example:")
    
    # Add domain-specific words (e.g., for movie reviews)
    movie_words = {
        'cinematography': 2.5,
        'plot': 1.0,
        'acting': 1.0,
        'screenplay': 1.5,
        'directing': 1.5,
        'boring': -2.5,
        'thrilling': 3.0,
        'captivating': 3.0
    }
    
    for word, score in movie_words.items():
        vader.add_to_lexicon(word, score)
    
    movie_review = "The cinematography was captivating but the plot was quite boring."
    movie_scores = vader.analyze_sentiment(movie_review)
    print(f"Movie review analysis:")
    print(f"Text: {movie_review}")
    print(f"Scores: {movie_scores}")
    
    # Compare before and after adding domain words
    basic_vader = VaderSentimentAnalyzer()  # Without domain words
    basic_scores = basic_vader.analyze_sentiment(movie_review)
    print(f"Without domain words: {basic_scores}")
    print(f"With domain words: {movie_scores}")
    
    print(f"\nImprovement in domain-specific analysis: {movie_scores['compound'] - basic_scores['compound']:.3f}")
