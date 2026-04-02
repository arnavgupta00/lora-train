# Domain-Augmented Text-to-SQL Training Data

## Overview
This dataset contains 400 high-quality text-to-SQL training examples across 4 diverse domain patterns.

**Generated**: 2024-04-02  
**Format**: JSONL (ChatML structure)  
**File**: `augment_domains.jsonl`  
**Size**: 319 KB

## Statistics

- **Total Examples**: 400
- **With JOINs**: 176 (44.0%)
- **SQLite Compatible**: ✓

### Distribution by Domain

| Domain | Examples | JOINs | Percentage |
|--------|----------|-------|------------|
| Chemistry/Molecular | 100 | 41 | 41.0% |
| Financial/Banking | 100 | 45 | 45.0% |
| Gaming/Characters | 100 | 47 | 47.0% |
| Medical/Clinical | 100 | 43 | 43.0% |

## Schemas

### 1. Chemistry/Molecular (SYNTHETIC)
Tables: `compounds`, `atoms`, `bonds`, `reactions`
- Compound properties (molecular weight, toxicity)
- Atomic composition and bonding patterns
- Chemical reactions with catalysts

### 2. Financial/Banking (SYNTHETIC)
Tables: `accounts`, `transactions`, `customers`, `loans`
- Customer accounts and balances
- Transaction history with categories
- Credit scores and loan management

### 3. Gaming/Characters (SYNTHETIC)
Tables: `heroes`, `abilities`, `teams`, `memberships`
- Hero attributes and power types
- Ability damage and cooldowns
- Team compositions and roles

### 4. Medical/Clinical (SYNTHETIC)
Tables: `patients`, `diagnoses`, `treatments`, `lab_results`
- Patient demographics and blood types
- Diagnosis severity levels
- Treatment outcomes and lab tests

## Query Complexity

- **Simple SELECTs**: ~65% (basic lookups, filters)
- **With Aggregations**: ~27% (COUNT, SUM, AVG, MAX, MIN)
- **Multi-table JOINs**: ~44% (2-3 table joins)
- **With Subqueries**: ~5% (nested queries)

## Format

Each line is a JSON object with ChatML structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Question\n\nSchema:\nCREATE TABLE..."
    },
    {
      "role": "assistant",
      "content": "SELECT ..."
    }
  ]
}
```

## Sample Examples

**Chemistry (Simple)**:
```
Q: List all toxic compounds
A: SELECT name FROM compounds WHERE is_toxic = 1
```

**Chemistry (JOIN)**:
```
Q: Find compounds containing oxygen atoms
A: SELECT DISTINCT c.name FROM compounds c JOIN atoms a 
   ON c.compound_id = a.compound_id WHERE a.element = 'O'
```

**Financial (JOIN)**:
```
Q: Calculate total transaction amount per customer
A: SELECT c.name, SUM(t.amount) as total FROM customers c 
   JOIN accounts a ON c.customer_id = a.customer_id 
   JOIN transactions t ON a.account_id = t.account_id 
   GROUP BY c.name
```

**Gaming (JOIN)**:
```
Q: Show hero names with their abilities
A: SELECT h.name, a.ability_name FROM heroes h 
   JOIN abilities a ON h.hero_id = a.hero_id
```

**Medical (JOIN)**:
```
Q: Find female patients with diabetes
A: SELECT p.name FROM patients p JOIN diagnoses d 
   ON p.patient_id = d.patient_id 
   WHERE p.gender = 'female' AND d.condition = 'diabetes'
```

## Notes

- All schemas are **SYNTHETIC** (not from BIRD or other benchmarks)
- Questions use natural English phrasing
- Full DDL schema included in each example
- SQLite-compatible syntax throughout
- Mix of query complexities for robust training

## Generation

Generated using Python script with carefully curated question-SQL pairs across each domain to ensure:
- Natural language variation
- Query pattern diversity
- Proper JOIN coverage
- Domain-specific terminology
