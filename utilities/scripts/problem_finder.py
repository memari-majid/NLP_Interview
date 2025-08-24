#!/usr/bin/env python3
"""
NLP Problem Finder - Quick navigation tool for interview prep
Usage: python problem_finder.py
"""

import os
import random
from typing import List, Dict, Tuple
from pathlib import Path

# Problem database with metadata
PROBLEMS = {
    "tokenization": {
        "path": "NLP/Tokenization/", 
        "difficulty": "easy",
        "time": 15,
        "companies": ["all"],
        "topics": ["preprocessing", "text-processing"],
        "related": ["bpe", "stop_words"]
    },
    "stop_words": {
        "path": "NLP/Stop_Word_Removal/",
        "difficulty": "easy", 
        "time": 15,
        "companies": ["amazon", "microsoft"],
        "topics": ["preprocessing", "search"],
        "related": ["tokenization", "tfidf"]
    },
    "stemming": {
        "path": "NLP/Stemming_Lemmatization/",
        "difficulty": "easy",
        "time": 20,
        "companies": ["google"],
        "topics": ["preprocessing", "normalization"],
        "related": ["tokenization"]
    },
    "bow": {
        "path": "NLP/BoW_Vectors/",
        "difficulty": "easy",
        "time": 20,
        "companies": ["meta", "apple"],
        "topics": ["vectorization", "features"],
        "related": ["tfidf", "word2vec"]
    },
    "similarity": {
        "path": "NLP/Similarity/",
        "difficulty": "easy",
        "time": 25,
        "companies": ["all"],
        "topics": ["similarity", "search"],
        "related": ["word2vec", "tfidf"]
    },
    "ngrams": {
        "path": "NLP/NGrams/",
        "difficulty": "easy",
        "time": 20,
        "companies": ["amazon"],
        "topics": ["features", "preprocessing"],
        "related": ["bow", "tokenization"]
    },
    "regex": {
        "path": "NLP/Regex_NLP/",
        "difficulty": "easy",
        "time": 25,
        "companies": ["startups"],
        "topics": ["extraction", "patterns"],
        "related": ["ner", "tokenization"]
    },
    "tfidf": {
        "path": "NLP/TFIDF/",
        "difficulty": "medium",
        "time": 25,
        "companies": ["google", "amazon"],
        "topics": ["vectorization", "search", "ranking"],
        "related": ["bow", "similarity", "word2vec"]
    },
    "word2vec": {
        "path": "NLP/Embeddings/",
        "difficulty": "medium",
        "time": 25,
        "companies": ["meta", "microsoft"],
        "topics": ["embeddings", "similarity"],
        "related": ["tfidf", "attention", "similarity"]
    },
    "classification": {
        "path": "NLP/Text_Classification/",
        "difficulty": "medium",
        "time": 25,
        "companies": ["all"],
        "topics": ["ml", "supervised", "classification"],
        "related": ["sentiment", "cnn", "bert"]
    },
    "pos": {
        "path": "NLP/POS_Tagging/",
        "difficulty": "medium",
        "time": 20,
        "companies": ["google"],
        "topics": ["tagging", "sequence-labeling"],
        "related": ["ner", "lstm"]
    },
    "ner": {
        "path": "NLP/NER/",
        "difficulty": "medium",
        "time": 25,
        "companies": ["amazon", "apple"],
        "topics": ["extraction", "tagging"],
        "related": ["pos", "regex", "bert"]
    },
    "sentiment": {
        "path": "NLP/Sentiment_Analysis/",
        "difficulty": "medium",
        "time": 20,
        "companies": ["startups", "amazon"],
        "topics": ["classification", "ml"],
        "related": ["classification", "vader"]
    },
    "perceptron": {
        "path": "NLP/Neural_Fundamentals/",
        "difficulty": "medium",
        "time": 25,
        "companies": ["meta"],
        "topics": ["neural", "fundamentals"],
        "related": ["cnn", "lstm"]
    },
    "attention": {
        "path": "NLP/Attention_Mechanisms/",
        "difficulty": "hard",
        "time": 30,
        "companies": ["openai", "google"],
        "topics": ["transformers", "attention"],
        "related": ["bert", "gpt", "transformers"]
    },
    "bert": {
        "path": "NLP/Transformers/",
        "difficulty": "hard",
        "time": 30,
        "companies": ["meta", "microsoft"],
        "topics": ["transformers", "fine-tuning"],
        "related": ["attention", "classification"]
    },
    "bpe": {
        "path": "NLP/Tokenization_Advanced/",
        "difficulty": "hard",
        "time": 30,
        "companies": ["openai", "anthropic"],
        "topics": ["tokenization", "subword"],
        "related": ["tokenization", "gpt"]
    },
    "gpt": {
        "path": "NLP/GPT_Implementation/",
        "difficulty": "hard",
        "time": 35,
        "companies": ["openai"],
        "topics": ["transformers", "generation"],
        "related": ["attention", "generation", "bpe"]
    },
    "lstm": {
        "path": "NLP/Sequence_Models/",
        "difficulty": "hard",
        "time": 30,
        "companies": ["google", "amazon"],
        "topics": ["rnn", "sequence"],
        "related": ["attention", "pos", "sentiment"]
    },
    "lda": {
        "path": "NLP/TopicModeling/",
        "difficulty": "hard",
        "time": 35,
        "companies": ["research"],
        "topics": ["unsupervised", "topics"],
        "related": ["bow", "tfidf"]
    },
    "cnn": {
        "path": "NLP/CNN_Text/",
        "difficulty": "hard",
        "time": 30,
        "companies": ["meta"],
        "topics": ["neural", "classification"],
        "related": ["classification", "lstm"]
    },
    "generation": {
        "path": "NLP/LLM_Fundamentals/",
        "difficulty": "hard",
        "time": 25,
        "companies": ["openai", "anthropic"],
        "topics": ["generation", "llm"],
        "related": ["gpt", "attention"]
    },
    "instruction": {
        "path": "NLP/Instruction_Tuning/",
        "difficulty": "hard",
        "time": 30,
        "companies": ["openai", "google"],
        "topics": ["fine-tuning", "llm"],
        "related": ["bert", "gpt"]
    }
}

class ProblemFinder:
    def __init__(self):
        self.problems = PROBLEMS
        
    def find_by_difficulty(self, difficulty: str) -> List[str]:
        """Find problems by difficulty level."""
        return [name for name, info in self.problems.items() 
                if info["difficulty"] == difficulty]
    
    def find_by_company(self, company: str) -> List[str]:
        """Find problems asked by specific company."""
        company = company.lower()
        return [name for name, info in self.problems.items()
                if company in info["companies"] or "all" in info["companies"]]
    
    def find_by_time(self, max_time: int) -> List[str]:
        """Find problems that fit in time limit."""
        return [name for name, info in self.problems.items()
                if info["time"] <= max_time]
    
    def find_by_topic(self, topic: str) -> List[str]:
        """Find problems by topic/concept."""
        topic = topic.lower()
        return [name for name, info in self.problems.items()
                if any(topic in t for t in info["topics"])]
    
    def get_related(self, problem_name: str) -> List[str]:
        """Get related problems to practice next."""
        if problem_name in self.problems:
            return self.problems[problem_name]["related"]
        return []
    
    def get_random_problem(self, criteria: Dict = None) -> str:
        """Get random problem matching criteria."""
        candidates = list(self.problems.keys())
        
        if criteria:
            if "difficulty" in criteria:
                candidates = [p for p in candidates 
                            if self.problems[p]["difficulty"] == criteria["difficulty"]]
            if "company" in criteria:
                candidates = [p for p in candidates
                            if criteria["company"] in self.problems[p]["companies"]]
            if "max_time" in criteria:
                candidates = [p for p in candidates
                            if self.problems[p]["time"] <= criteria["max_time"]]
        
        return random.choice(candidates) if candidates else None
    
    def get_study_session(self, time_minutes: int) -> List[Tuple[str, int]]:
        """Generate study session fitting in time limit."""
        session = []
        remaining_time = time_minutes
        used_problems = set()
        
        # Try to fit problems optimally
        while remaining_time >= 15:  # Minimum problem time
            candidates = [
                (name, info["time"]) 
                for name, info in self.problems.items()
                if info["time"] <= remaining_time and name not in used_problems
            ]
            
            if not candidates:
                break
                
            # Pick problem that best uses remaining time
            candidates.sort(key=lambda x: abs(remaining_time - x[1]))
            chosen = candidates[0]
            
            session.append(chosen)
            used_problems.add(chosen[0])
            remaining_time -= chosen[1]
        
        return session
    
    def display_problem_info(self, problem_name: str):
        """Display detailed problem information."""
        if problem_name not in self.problems:
            print(f"Problem '{problem_name}' not found")
            return
            
        info = self.problems[problem_name]
        print(f"\n{'='*50}")
        print(f"Problem: {problem_name.upper()}")
        print(f"{'='*50}")
        print(f"Path: {info['path']}")
        print(f"Difficulty: {info['difficulty'].capitalize()}")
        print(f"Time: {info['time']} minutes")
        print(f"Companies: {', '.join(info['companies'])}")
        print(f"Topics: {', '.join(info['topics'])}")
        print(f"Related: {', '.join(info['related'])}")
        print(f"\nFiles:")
        print(f"  - Problem: {info['path']}{problem_name}_problem.md")
        print(f"  - Solution: {info['path']}{problem_name}_solution.py")

def interactive_menu():
    """Interactive problem finder menu."""
    finder = ProblemFinder()
    
    while True:
        print("\n" + "="*50)
        print("NLP INTERVIEW PROBLEM FINDER")
        print("="*50)
        print("1. Find by difficulty (easy/medium/hard)")
        print("2. Find by company")
        print("3. Find by time limit")
        print("4. Find by topic")
        print("5. Get random problem")
        print("6. Generate study session")
        print("7. Show problem details")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == "1":
            diff = input("Enter difficulty (easy/medium/hard): ").lower()
            problems = finder.find_by_difficulty(diff)
            print(f"\n{diff.capitalize()} problems: {', '.join(problems)}")
            
        elif choice == "2":
            print("Companies: openai, google, meta, amazon, microsoft, apple")
            company = input("Enter company: ").lower()
            problems = finder.find_by_company(company)
            print(f"\n{company.capitalize()} problems: {', '.join(problems)}")
            
        elif choice == "3":
            time = int(input("Enter max time in minutes: "))
            problems = finder.find_by_time(time)
            print(f"\nProblems ≤ {time} min: {', '.join(problems)}")
            
        elif choice == "4":
            topic = input("Enter topic (e.g., transformers, classification): ")
            problems = finder.find_by_topic(topic)
            print(f"\n{topic} problems: {', '.join(problems)}")
            
        elif choice == "5":
            print("\nRandom problem criteria (press Enter to skip):")
            diff = input("Difficulty (easy/medium/hard): ").lower() or None
            company = input("Company: ").lower() or None
            max_time = input("Max time (minutes): ")
            max_time = int(max_time) if max_time else None
            
            criteria = {}
            if diff: criteria["difficulty"] = diff
            if company: criteria["company"] = company
            if max_time: criteria["max_time"] = max_time
            
            problem = finder.get_random_problem(criteria)
            if problem:
                finder.display_problem_info(problem)
            else:
                print("No matching problems found")
                
        elif choice == "6":
            time = int(input("Enter session time in minutes: "))
            session = finder.get_study_session(time)
            
            print(f"\nStudy Session ({time} minutes):")
            print("-" * 30)
            total_time = 0
            for prob, prob_time in session:
                print(f"{prob:20} - {prob_time} min")
                total_time += prob_time
            print("-" * 30)
            print(f"Total: {total_time} minutes")
            print(f"Unused: {time - total_time} minutes")
            
        elif choice == "7":
            problem = input("Enter problem name: ").lower()
            finder.display_problem_info(problem)
            
        elif choice == "8":
            print("\nHappy studying! 🚀")
            break
            
        else:
            print("Invalid option")

if __name__ == "__main__":
    print("Welcome to NLP Interview Problem Finder!")
    print("This tool helps you efficiently navigate the problem bank.")
    interactive_menu()
