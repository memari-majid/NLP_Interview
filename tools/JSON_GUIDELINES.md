# JSON Guidelines for Flashcard Creation

## Common JSON Issues and Solutions

### ❌ **Invalid Escape Sequences**
**Problem:** Using backslashes incorrectly in JSON strings
```json
"Use re.split with pattern \\s"     // ❌ Invalid - \s is not valid JSON escape
"Use re.split with pattern \\\\s"   // ✅ Correct - double backslash for literal \s
```

**Problem:** Escaping single quotes unnecessarily
```json
"Use pattern [,.:;?\\'_!\")         // ❌ Invalid - \' is not valid JSON escape  
"Use pattern [,.:;?'_!\")           // ✅ Correct - single quotes don't need escaping
```

### ✅ **Valid JSON Escape Sequences**
Only these are valid in JSON strings:
- `\"` - Double quote
- `\\` - Backslash  
- `\/` - Forward slash (optional)
- `\b` - Backspace
- `\f` - Form feed
- `\n` - Newline
- `\r` - Carriage return
- `\t` - Tab
- `\uXXXX` - Unicode character

### 🔧 **Common Fixes**

**Code/Regex patterns:**
```json
// ❌ Wrong
"pattern": "\\s+"
"regex": "\\d{3}\\-\\d{3}\\-\\d{4}"

// ✅ Correct  
"pattern": "\\\\s+"
"regex": "\\\\d{3}\\\\-\\\\d{3}\\\\-\\\\d{4}"
```

**Quotes in content:**
```json
// ❌ Wrong
"text": "He said \"Hello\" and she said \'Hi\'"

// ✅ Correct
"text": "He said \"Hello\" and she said 'Hi'"
```

## Best Practices

1. **Always validate JSON** before committing:
   ```bash
   python3 -m json.tool your_file.json > /dev/null
   ```

2. **Use proper escaping** for code examples:
   - Single `\\` for literal backslash in final output
   - Double `\\\\` in JSON source to produce single `\\`

3. **Test with the conversion script** - it will show detailed error messages

4. **Common patterns to watch:**
   - Regex patterns with `\s`, `\d`, `\w`
   - File paths with backslashes  
   - Code snippets with special characters

## Error Messages Explained

When you see:
```
json.decoder.JSONDecodeError: Invalid \escape: line 179 column 348
```

This means:
- **Line 179**: The JSON line with the error
- **Column 348**: Character position in that line  
- **Invalid \escape**: An invalid backslash sequence was found

## Script Features

The updated conversion script now:
- ✅ **Validates JSON** and shows exact error locations
- ✅ **Skips invalid files** instead of crashing
- ✅ **Auto-detects directories** with JSON files
- ✅ **Handles any directory** structure you create
- ✅ **Provides detailed error messages** for troubleshooting

## Usage

1. Create any directory under `data/source/flashcards/YOUR_TOPIC/`
2. Add numbered JSON files (1.json, 2.json, etc.)
3. Run the conversion script and choose option 4
4. Select your directory and name your collection
5. Get a combined APKG file ready for Anki import!