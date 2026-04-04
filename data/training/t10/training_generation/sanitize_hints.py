#!/usr/bin/env python3
"""Sanitize junky hints in T10 dataset."""

import json
import re

def is_junky_hint(hint_text):
    """Check if a hint is junky/empty and should be replaced with None."""
    if not hint_text or hint_text.strip() == "":
        return True
    
    # Remove whitespace for checking
    cleaned = hint_text.strip()
    
    # Patterns that indicate junky hints
    junky_patterns = [
        r'^FALSE;?$',
        r'^TRUE;?$',
        r'^;+$',
        r'^This is not;?$',
        r'^None;?$',
        r'^N/A;?$',
        r'^\.+$',
        r'^-+$',
        r'^_+$',
        r'^\s*$',
    ]
    
    for pattern in junky_patterns:
        if re.match(pattern, cleaned, re.IGNORECASE):
            return True
    
    # If it's very short (< 5 chars) and doesn't contain alphanumeric, probably junk
    if len(cleaned) < 5 and not re.search(r'[a-zA-Z0-9]', cleaned):
        return True
    
    return False

def sanitize_example(example, example_id):
    """Sanitize hints in an example. Returns (sanitized_example, was_changed, old_hint)."""
    content = example["messages"][1]["content"]
    
    # Extract the hints section
    hints_match = re.search(r'Hints:\n(.*?)\n\nQuestion:', content, re.DOTALL)
    if not hints_match:
        return example, False, None
    
    current_hint = hints_match.group(1)
    
    # Skip if already "None"
    if current_hint.strip() == "None":
        return example, False, None
    
    # Check if junky
    if is_junky_hint(current_hint):
        # Replace with None
        new_content = content.replace(f"Hints:\n{current_hint}\n\n", "Hints:\nNone\n\n")
        example["messages"][1]["content"] = new_content
        return example, True, current_hint
    
    return example, False, None

# Process train
train_sanitized = []
train_count = 0
with open("train_t10.jsonl") as f_in, open("train_t10_sanitized.jsonl", "w") as f_out:
    for idx, line in enumerate(f_in):
        example = json.loads(line)
        example_id = f"train_{idx}"
        
        sanitized, was_changed, old_hint = sanitize_example(example, example_id)
        
        if was_changed:
            train_sanitized.append({
                "example_id": example_id,
                "old_hint": old_hint,
                "new_hint": "None"
            })
            train_count += 1
        
        f_out.write(json.dumps(sanitized, ensure_ascii=False) + "\n")

print(f"Train: {train_count} hints sanitized")

# Process dev
dev_sanitized = []
dev_count = 0
with open("dev_t10.jsonl") as f_in, open("dev_t10_sanitized.jsonl", "w") as f_out:
    for idx, line in enumerate(f_in):
        example = json.loads(line)
        example_id = f"dev_{idx}"
        
        sanitized, was_changed, old_hint = sanitize_example(example, example_id)
        
        if was_changed:
            dev_sanitized.append({
                "example_id": example_id,
                "old_hint": old_hint,
                "new_hint": "None"
            })
            dev_count += 1
        
        f_out.write(json.dumps(sanitized, ensure_ascii=False) + "\n")

print(f"Dev: {dev_count} hints sanitized")

# Save sanitization log
sanitization_log = {
    "train_sanitized": train_sanitized,
    "dev_sanitized": dev_sanitized,
    "summary": {
        "train_count": train_count,
        "dev_count": dev_count,
        "total": train_count + dev_count
    }
}

with open("sanitization_log.json", "w") as f:
    json.dump(sanitization_log, f, indent=2, ensure_ascii=False)

print(f"\nTotal sanitized: {train_count + dev_count}")
print("Sanitization log saved to: sanitization_log.json")

# Show first few examples
if train_sanitized or dev_sanitized:
    print("\n=== Sample Sanitized Hints ===")
    for item in (train_sanitized + dev_sanitized)[:10]:
        print(f"\n{item['example_id']}:")
        print(f"  Old: {repr(item['old_hint'])}")
        print(f"  New: {item['new_hint']}")
