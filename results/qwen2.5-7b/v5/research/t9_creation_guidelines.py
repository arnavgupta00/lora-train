#!/usr/bin/env python3
"""
T9 Dataset Creation Guidelines

This document outlines how to create a T9 dataset that better matches BIRD distribution.
"""

import json
from collections import defaultdict

# TARGET DISTRIBUTION for T9
# Based on BIRD benchmark analysis

TARGET_DISTRIBUTION = {
    # Critical patterns (must be close to BIRD)
    'join': {
        'bird_pct': 74.3,
        'target_pct': 70.0,  # Aim for 70%+
        'priority': 'CRITICAL',
        'action': 'INCREASE - currently at 48% in T8'
    },
    
    # Over-represented in training (need to reduce)
    'order': {
        'bird_pct': 24.3,
        'target_pct': 30.0,  # Allow slight over-representation
        'priority': 'HIGH',
        'action': 'REDUCE - currently at 47% in T8'
    },
    'limit': {
        'bird_pct': 18.5,
        'target_pct': 25.0,  # Important for "top N" queries
        'priority': 'HIGH',
        'action': 'REDUCE - currently at 32% in T8'
    },
    'case': {
        'bird_pct': 13.3,
        'target_pct': 15.0,  # Slight over to learn pattern
        'priority': 'HIGH',
        'action': 'REDUCE - T8 has 23%, caused regression'
    },
    
    # Under-represented (need to increase)
    'distinct': {
        'bird_pct': 22.3,
        'target_pct': 20.0,
        'priority': 'HIGH',
        'action': 'INCREASE - currently at 10.9% in T8'
    },
    'subquery': {
        'bird_pct': 15.2,
        'target_pct': 15.0,
        'priority': 'MEDIUM',
        'action': 'INCREASE - currently at 9.1% in T8'
    },
    
    # Good in T8 (maintain)
    'cte': {
        'bird_pct': 6.6,
        'target_pct': 6.0,
        'priority': 'LOW',
        'action': 'MAINTAIN - T8 at 4.4% is close enough'
    },
    'window': {
        'bird_pct': 4.4,
        'target_pct': 5.0,
        'priority': 'LOW',
        'action': 'MAINTAIN - T8 at 4.8% is good'
    },
    'group_by': {
        'bird_pct': 11.9,
        'target_pct': 12.0,
        'priority': 'LOW',
        'action': 'MAINTAIN - T8 at 14.6% is acceptable'
    },
}

# SPECIAL REQUIREMENTS

SPECIAL_REQUIREMENTS = {
    'backtick_columns': {
        'current_pct': 5.0,
        'target_pct': 3.0,  # Increase absolute count, but maintain ratio
        'target_count': 500,  # Need 500+ examples (currently ~80)
        'priority': 'CRITICAL',
        'note': 'california_schools-style columns are causing 18% accuracy'
    },
    'schema_format': {
        'current': '100% DDL',
        'target': '100% DDL',
        'action': 'MAINTAIN - DDL format matches BIRD eval'
    },
}

# RECOMMENDED DATASET COMPOSITION

RECOMMENDED_COMPOSITION = """
## T9 Dataset Composition Recommendations

### Base Data Sources (estimated 18,000 examples)
1. **BIRD Training Set**: ~9,400 examples
   - Keep ALL BIRD training examples
   - These have correct distribution by definition
   
2. **Spider Dataset**: ~7,000 examples  
   - Filter for SQLite compatibility
   - Good coverage of standard patterns
   
3. **Custom T3 Data**: ~1,600 examples
   - Keep only high-quality examples
   - Remove any with distribution outliers

### Augmentation Strategy

#### 1. JOIN Upsampling (+3,000 examples)
Generate synthetic JOIN-heavy examples:
- Multi-table JOINs (3+ tables)
- Various JOIN types (INNER, LEFT, RIGHT)
- Complex foreign key relationships

#### 2. DISTINCT/SUBQUERY Upsampling (+1,000 examples)
Currently under-represented:
- Add nested subqueries in WHERE
- Add DISTINCT with aggregations

#### 3. California Schools Style (+420 examples)
Backtick column examples:
- 80 current → 500 target
- Include `Column With (Parens)` format
- Include spaces, special chars in column names

### Pattern Balancing

To achieve target distribution from 22,000 base examples:

| Pattern | Current Count | Target Count | Action |
|---------|--------------|--------------|--------|
| JOIN | 11,068 | 15,400 | +4,332 (add JOIN-heavy examples) |
| ORDER | 10,736 | 6,600 | -4,136 (filter out simple ORDER BY) |
| CASE | 5,302 | 3,300 | -2,002 (reduce CASE upsampling) |
| DISTINCT | 2,483 | 4,400 | +1,917 (add DISTINCT examples) |
| SUBQUERY | 2,065 | 3,300 | +1,235 (add nested queries) |
| LIMIT | 7,396 | 5,500 | -1,896 (reduce simple LIMIT) |

### Final T9 Stats (Target)

- Total examples: ~22,000
- JOIN: 70%
- ORDER: 30%
- DISTINCT: 20%
- LIMIT: 25%
- SUBQUERY: 15%
- CASE: 15%
- CTE: 6%
- WINDOW: 5%
- Backtick columns: 500+ examples
"""

def print_guidelines():
    print("=" * 60)
    print("T9 DATASET CREATION GUIDELINES")
    print("=" * 60)
    
    print("\n## Target Pattern Distribution\n")
    print(f"{'Pattern':<15} {'BIRD %':<10} {'Target %':<10} {'Priority':<10} {'Action'}")
    print("-" * 70)
    
    for pattern, config in sorted(TARGET_DISTRIBUTION.items(), 
                                   key=lambda x: -x[1]['bird_pct']):
        print(f"{pattern:<15} {config['bird_pct']:<10.1f} {config['target_pct']:<10.1f} "
              f"{config['priority']:<10} {config['action'][:30]}...")
    
    print("\n## Special Requirements\n")
    for req, config in SPECIAL_REQUIREMENTS.items():
        print(f"- **{req}**: {config.get('note', config.get('action', ''))}")
    
    print(RECOMMENDED_COMPOSITION)


if __name__ == '__main__':
    print_guidelines()
