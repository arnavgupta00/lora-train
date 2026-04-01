import json

# Process base file
with open('predictions.test.base.jsonl', 'r') as f:
    base_matches = []
    for line in f:
        entry = json.loads(line)
        if entry.get('exact_match') or entry.get('execution_match'):
            base_matches.append(entry)

# Process lora file
with open('predictions.test.lora.jsonl', 'r') as f:
    lora_matches = []
    for line in f:
        entry = json.loads(line)
        if entry.get('exact_match') or entry.get('execution_match'):
            lora_matches.append(entry)

# Save results
with open('matches_base.jsonl', 'w') as f:
    for entry in base_matches:
        f.write(json.dumps(entry) + '\n')

with open('matches_lora.jsonl', 'w') as f:
    for entry in lora_matches:
        f.write(json.dumps(entry) + '\n')

print(f"Base matches: {len(base_matches)}")
print(f"Lora matches: {len(lora_matches)}")
print(f"\nBase file breakdown:")
base_exact = sum(1 for e in base_matches if e.get('exact_match'))
base_exec = sum(1 for e in base_matches if e.get('execution_match'))
base_both = sum(1 for e in base_matches if e.get('exact_match') and e.get('execution_match'))
print(f"  - exact_match only: {base_exact - base_both}")
print(f"  - execution_match only: {base_exec - base_both}")
print(f"  - both: {base_both}")

print(f"\nLora file breakdown:")
lora_exact = sum(1 for e in lora_matches if e.get('exact_match'))
lora_exec = sum(1 for e in lora_matches if e.get('execution_match'))
lora_both = sum(1 for e in lora_matches if e.get('exact_match') and e.get('execution_match'))
print(f"  - exact_match only: {lora_exact - lora_both}")
print(f"  - execution_match only: {lora_exec - lora_both}")
print(f"  - both: {lora_both}")
