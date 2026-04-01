# T8 Dataset - Eval File Format Explanation

## Why Does eval/bird_dev.jsonl Look Different?

**TL;DR:** The eval file has placeholder schemas because the **actual schema is extracted from SQLite databases during evaluation**. This is the correct format!

---

## Format Comparison

### Training Files (train.jsonl, dev.jsonl)
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert SQL assistant..."},
    {"role": "user", "content": "Schema:\nCREATE TABLE users (...)\n\nQuestion: ..."},
    {"role": "assistant", "content": "SELECT * FROM users..."}
  ],
  "source": "nl2sql_dataset_service"
}
```

**Purpose:** Used by `train_lora.py` for training - contains complete prompt-response pairs.

---

### Eval File (eval/bird_dev.jsonl)
```json
{
  "db_id": "california_schools",
  "question_id": 0,
  "question": "What is the highest free meal rate...",
  "evidence": "Free meal rate = ...",
  "gold_sql": "SELECT MAX(...) FROM frpm...",
  "difficulty": "challenging",
  "messages": [
    {"role": "system", "content": "You are an expert SQL assistant..."},
    {"role": "user", "content": "Schema:\n[Schema will be injected during evaluation]\n\nQuestion: ..."}
  ]
}
```

**Purpose:** Metadata for BIRD benchmark evaluation - schema is extracted at runtime.

---

## How Evaluation Works

The `evaluation/run_bird_eval.sh` script does this:

```python
# 1. Read metadata from bird_dev.jsonl
data = json.load(bird_dev.jsonl)
db_id = data['db_id']              # "california_schools"
question = data['question']        # "What is the highest..."
gold_sql = data['gold_sql']        # Ground truth for scoring

# 2. Find SQLite database
db_path = f"bird_eval/dev_databases/{db_id}/{db_id}.sqlite"

# 3. Extract REAL schema from database
conn = sqlite3.connect(db_path)
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
schema = "\n".join(row[0] for row in cursor.fetchall())
# Result: "CREATE TABLE frpm (CDSCode TEXT, ...)\nCREATE TABLE schools (...)"

# 4. Build actual prompt
user_content = f"Schema:\n{schema}\n\nQuestion: {question}"
messages = [
    {"role": "system", "content": "You are an expert SQL assistant..."},
    {"role": "user", "content": user_content}
]

# 5. Generate prediction
prompt = tokenizer.apply_chat_template(messages, ...)
prediction = model.generate(prompt)

# 6. Compare to gold_sql
execute(prediction) == execute(gold_sql)  # Execution accuracy
```

---

## Why This Design?

1. **SQLite databases contain the actual schema** with:
   - Exact column names (including backticks)
   - Data types
   - Foreign keys
   - Indexes

2. **Schema extraction matches training** - The eval v3 script uses the same DDL format that t7/t8 training data used

3. **Prevents hardcoding** - If schemas were in the JSONL, they could become outdated if databases are updated

4. **Standard BIRD format** - This matches how the official BIRD benchmark works

---

## Verification

Check that the eval file has all required fields:

```bash
# All 1,534 examples have required fields
cat data/training/t8/eval/bird_dev.jsonl | jq -r 'select(.db_id and .question and .gold_sql) | "ok"' | wc -l
# Output: 1534

# All 11 BIRD dev databases present
cat data/training/t8/eval/bird_dev.jsonl | jq -r '.db_id' | sort -u
# Output: california_schools, card_games, codebase_community, ...
```

---

## Summary

✅ **The eval file format is correct!**

- The `[Schema will be injected during evaluation]` placeholder is intentional
- The `messages` field is metadata (not used by eval script)
- The eval script extracts **real DDL schemas from SQLite databases** at runtime
- This ensures schema format matches training (CREATE TABLE statements)
