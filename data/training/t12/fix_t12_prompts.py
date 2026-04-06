#!/usr/bin/env python3
"""
Fix T12 additions to match the T12 prompt contract.

Issues to fix:
1. System prompt has \n instead of actual newlines
2. Some examples missing "Hints:" section in user prompt
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from t12_utils import T12_SYSTEM_PROMPT, validate_t12_messages


def fix_system_prompt(system_prompt):
    """Fix system prompt to have actual newlines."""
    # Replace \n with actual newlines
    fixed = system_prompt.replace('\\n', '\n')
    # Ensure it matches T12 exactly
    if fixed.strip() != T12_SYSTEM_PROMPT.strip():
        # Use the canonical version
        return T12_SYSTEM_PROMPT
    return fixed


def fix_user_prompt(user_prompt):
    """Add Hints section if missing."""
    if 'Hints:' in user_prompt:
        return user_prompt
    
    # Find Question: section
    if 'Question:' not in user_prompt:
        raise ValueError("User prompt missing Question: section")
    
    # Insert "Hints:\nNone\n\n" before Question:
    parts = user_prompt.split('Question:')
    if len(parts) != 2:
        raise ValueError(f"Unexpected Question: format in user prompt")
    
    fixed = parts[0].rstrip() + '\n\nHints:\nNone\n\nQuestion:' + parts[1]
    return fixed


def fix_t12_training_file(input_file, output_file):
    """Fix T12 training file."""
    backbone_count = 14034
    fixed_count = 0
    error_count = 0
    errors = []
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    print()
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            
            try:
                ex = json.loads(line)
                messages = ex.get('messages', [])
                
                # Only fix T12 additions (after backbone)
                if i >= backbone_count:
                    # Fix system prompt
                    if len(messages) > 0 and messages[0]['role'] == 'system':
                        messages[0]['content'] = fix_system_prompt(messages[0]['content'])
                    
                    # Fix user prompt
                    if len(messages) > 1 and messages[1]['role'] == 'user':
                        try:
                            messages[1]['content'] = fix_user_prompt(messages[1]['content'])
                            fixed_count += 1
                        except ValueError as e:
                            error_count += 1
                            errors.append(f"Line {i+1}: {e}")
                            if len(errors) <= 5:
                                print(f"ERROR on line {i+1}: {e}")
                    
                    # Validate
                    is_valid, errs = validate_t12_messages(messages, strict=False)
                    if not is_valid:
                        error_count += 1
                        if len(errors) <= 5:
                            print(f"WARNING: Line {i+1} still has validation errors: {errs}")
                
                # Write back
                f_out.write(json.dumps(ex) + '\n')
                
                if (i + 1) % 2000 == 0:
                    print(f"Processed {i + 1} examples...")
            
            except Exception as e:
                error_count += 1
                errors.append(f"Line {i+1}: {e}")
                if len(errors) <= 5:
                    print(f"FATAL ERROR on line {i+1}: {e}")
                # Write original line
                f_out.write(line)
    
    print()
    print("=" * 60)
    print(f"Total examples processed: {i + 1}")
    print(f"T12 additions fixed: {fixed_count}")
    print(f"Errors encountered: {error_count}")
    print("=" * 60)
    
    if error_count > 0:
        print(f"\nFirst {min(5, len(errors))} errors:")
        for err in errors[:5]:
            print(f"  {err}")
    
    return error_count == 0


if __name__ == "__main__":
    input_file = Path(__file__).parent / "train_t12.jsonl"
    output_file = Path(__file__).parent / "train_t12_fixed.jsonl"
    
    success = fix_t12_training_file(input_file, output_file)
    
    if success:
        print("\n✓ All fixes applied successfully!")
        print(f"\nTo use the fixed file, run:")
        print(f"  mv {output_file} {input_file}")
    else:
        print("\n✗ Some errors occurred. Review the output above.")
        sys.exit(1)
