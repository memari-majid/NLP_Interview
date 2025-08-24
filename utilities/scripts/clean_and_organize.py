#!/usr/bin/env python3
"""
Repository cleanup and organization script for NLP interview prep.
Ensures optimal Anki integration and clean structure.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set

def check_problem_solution_pairs() -> Dict[str, List[str]]:
    """Check for missing problem/solution pairs."""
    issues = {}
    nlp_dir = Path("NLP")
    
    for topic_dir in nlp_dir.iterdir():
        if not topic_dir.is_dir():
            continue
            
        problems = list(topic_dir.glob("*_problem.md"))
        solutions = list(topic_dir.glob("*_solution.py"))
        
        # Check for orphaned files
        problem_names = {p.stem.replace('_problem', '') for p in problems}
        solution_names = {s.stem.replace('_solution', '') for s in solutions}
        
        missing_solutions = problem_names - solution_names
        missing_problems = solution_names - problem_names
        
        if missing_solutions or missing_problems:
            issues[topic_dir.name] = []
            for name in missing_solutions:
                issues[topic_dir.name].append(f"Missing solution: {name}_solution.py")
            for name in missing_problems:
                issues[topic_dir.name].append(f"Missing problem: {name}_problem.md")
    
    return issues

def validate_anki_decks() -> Dict[str, str]:
    """Validate Anki deck JSON files."""
    validation_results = {}
    
    deck_files = [
        Path("anki_deck/NLP_Interview_Deck.json"),
        Path("anki_deck_optimized/NLP_Interview_Optimized.json")
    ]
    
    for deck_file in deck_files:
        if deck_file.exists():
            try:
                with open(deck_file, 'r') as f:
                    data = json.load(f)
                    if "__type__" in data and data["__type__"] == "Deck":
                        note_count = len(data.get("notes", []))
                        validation_results[str(deck_file)] = f"Valid (contains {note_count} notes)"
                    else:
                        validation_results[str(deck_file)] = "Invalid deck structure"
            except json.JSONDecodeError as e:
                validation_results[str(deck_file)] = f"Invalid JSON: {e}"
        else:
            validation_results[str(deck_file)] = "File not found"
    
    return validation_results

def check_code_quality() -> Dict[str, List[str]]:
    """Check solution files for quality indicators."""
    quality_issues = {}
    
    for solution_file in Path("NLP").rglob("*_solution.py"):
        issues = []
        
        with open(solution_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            
            # Check for complexity comments
            if not any('O(' in line for line in lines):
                issues.append("Missing complexity analysis")
            
            # Check for docstrings
            if '"""' not in content and "'''" not in content:
                issues.append("Missing docstrings")
            
            # Check for edge case handling
            if not any(keyword in content.lower() for keyword in ['edge', 'handle', 'check', 'validate']):
                issues.append("No explicit edge case handling")
            
            # Check for interview tips
            if not any(marker in content for marker in ['Key:', 'Interview:', 'Follow-up:']):
                issues.append("Missing interview tips/key concepts")
            
            # Check file size (for Anki optimization)
            if len(lines) > 200:
                issues.append(f"File too long ({len(lines)} lines) - consider breaking into smaller functions")
        
        if issues:
            quality_issues[str(solution_file)] = issues
    
    return quality_issues

def organize_structure() -> Dict[str, str]:
    """Suggest structural improvements."""
    suggestions = {}
    
    # Check for README in each topic directory
    for topic_dir in Path("NLP").iterdir():
        if topic_dir.is_dir():
            readme_path = topic_dir / "README.md"
            if not readme_path.exists():
                suggestions[str(topic_dir)] = "Consider adding README.md with topic overview"
    
    # Check for test files
    test_count = len(list(Path(".").rglob("test_*.py")))
    if test_count == 0:
        suggestions["tests"] = "No test files found - consider adding unit tests"
    
    return suggestions

def generate_report():
    """Generate comprehensive cleanup report."""
    print("=" * 60)
    print("NLP INTERVIEW REPO CLEANUP & OPTIMIZATION REPORT")
    print("=" * 60)
    
    # 1. Check problem/solution pairs
    print("\n📁 Problem/Solution Pairing:")
    print("-" * 40)
    pair_issues = check_problem_solution_pairs()
    if pair_issues:
        for topic, issues in pair_issues.items():
            print(f"❌ {topic}:")
            for issue in issues:
                print(f"   - {issue}")
    else:
        print("✅ All problems have matching solutions")
    
    # 2. Validate Anki decks
    print("\n🃏 Anki Deck Validation:")
    print("-" * 40)
    deck_validation = validate_anki_decks()
    for deck, status in deck_validation.items():
        symbol = "✅" if "Valid" in status else "❌"
        print(f"{symbol} {deck}: {status}")
    
    # 3. Check code quality
    print("\n📊 Code Quality Issues:")
    print("-" * 40)
    quality_issues = check_code_quality()
    if quality_issues:
        for file, issues in list(quality_issues.items())[:5]:  # Show first 5
            print(f"⚠️  {Path(file).name}:")
            for issue in issues:
                print(f"   - {issue}")
        if len(quality_issues) > 5:
            print(f"   ... and {len(quality_issues) - 5} more files with issues")
    else:
        print("✅ All solution files meet quality standards")
    
    # 4. Structure suggestions
    print("\n🏗️  Structural Improvements:")
    print("-" * 40)
    suggestions = organize_structure()
    if suggestions:
        for area, suggestion in suggestions.items():
            print(f"💡 {area}: {suggestion}")
    else:
        print("✅ Repository structure is well-organized")
    
    # 5. Anki optimization tips
    print("\n🎯 Anki Optimization Recommendations:")
    print("-" * 40)
    print("1. Use convert_to_anki_optimized.py for bite-sized cards")
    print("2. Each solution should have 5-8 focused functions")
    print("3. Add KEY:, FORMULA:, and COMPLEXITY: comments")
    print("4. Keep functions under 20 lines for better cards")
    print("5. Use the Example_Anki_Refactor as a template")
    
    # 6. Quick stats
    print("\n📈 Repository Statistics:")
    print("-" * 40)
    problem_count = len(list(Path("NLP").rglob("*_problem.md")))
    solution_count = len(list(Path("NLP").rglob("*_solution.py")))
    topic_count = len([d for d in Path("NLP").iterdir() if d.is_dir()])
    
    print(f"Topics: {topic_count}")
    print(f"Problems: {problem_count}")
    print(f"Solutions: {solution_count}")
    print(f"Anki converters: 2 (standard + optimized)")
    
    print("\n" + "=" * 60)
    print("Run 'python convert_to_anki_optimized.py' to generate cards")
    print("=" * 60)

if __name__ == "__main__":
    generate_report()