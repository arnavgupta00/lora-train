"""
Contrastive Example Generator

Generates archetype-driven contrastive repair pairs for high-value failure patterns.
These examples target specific, commonly confused patterns.

Patterns include:
- Rank by X, return Y
- ID vs name confusion
- COUNT(*) vs COUNT(DISTINCT)
- Wrong denominator/cohort
- Table-family confusion
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .config import BuilderConfig
from .metadata import (
    ContaminationSource,
    ExampleMetadata,
    Pool,
    RepairExample,
    SourceType,
)
from .taxonomy import FailureType, TargetFailureMode


@dataclass
class ContrastivePattern:
    """Defines a contrastive pattern with correct/incorrect variants."""
    pattern_id: str
    name: str
    failure_type: FailureType
    target_failure_mode: TargetFailureMode
    description: str
    # SQL templates - {placeholders} for substitution
    broken_template: str
    corrected_template: str
    # Required context
    required_tables: List[str] = field(default_factory=list)
    required_columns: Dict[str, List[str]] = field(default_factory=dict)


# Pre-defined contrastive patterns based on failure archetypes
CONTRASTIVE_PATTERNS: List[ContrastivePattern] = [
    # Return field errors
    ContrastivePattern(
        pattern_id="rank_return_wrong_field",
        name="Rank by X, Return Wrong Field",
        failure_type=FailureType.WRONG_RETURN_FIELD,
        target_failure_mode=TargetFailureMode.RETURN_FIELD_MISMATCH,
        description="Query sorts/ranks by one field but incorrectly returns the same field instead of the requested field",
        broken_template="SELECT {rank_field} FROM {table} ORDER BY {rank_field} {direction} LIMIT {limit}",
        corrected_template="SELECT {return_field} FROM {table} ORDER BY {rank_field} {direction} LIMIT {limit}",
    ),
    ContrastivePattern(
        pattern_id="id_vs_name",
        name="Return ID Instead of Name",
        failure_type=FailureType.WRONG_RETURN_FIELD,
        target_failure_mode=TargetFailureMode.RETURN_FIELD_MISMATCH,
        description="Returns the ID column when the name/title column was requested",
        broken_template="SELECT {id_field} FROM {table} WHERE {condition}",
        corrected_template="SELECT {name_field} FROM {table} WHERE {condition}",
    ),
    # Count granularity errors
    ContrastivePattern(
        pattern_id="count_vs_count_distinct",
        name="COUNT(*) vs COUNT(DISTINCT)",
        failure_type=FailureType.WRONG_COUNT_GRANULARITY,
        target_failure_mode=TargetFailureMode.WRONG_COUNT_GRANULARITY,
        description="Uses COUNT(*) when COUNT(DISTINCT) is needed for unique counting",
        broken_template="SELECT COUNT(*) FROM {table} WHERE {condition}",
        corrected_template="SELECT COUNT(DISTINCT {count_field}) FROM {table} WHERE {condition}",
    ),
    ContrastivePattern(
        pattern_id="count_distinct_wrong_field",
        name="COUNT(DISTINCT wrong_field)",
        failure_type=FailureType.WRONG_COUNT_GRANULARITY,
        target_failure_mode=TargetFailureMode.WRONG_COUNT_GRANULARITY,
        description="Counts distinct on wrong column",
        broken_template="SELECT COUNT(DISTINCT {wrong_field}) FROM {table}",
        corrected_template="SELECT COUNT(DISTINCT {correct_field}) FROM {table}",
    ),
    # Denominator errors
    ContrastivePattern(
        pattern_id="percentage_wrong_denominator",
        name="Wrong Percentage Denominator",
        failure_type=FailureType.WRONG_DENOMINATOR,
        target_failure_mode=TargetFailureMode.WRONG_DENOMINATOR,
        description="Uses filtered count as denominator instead of total count",
        broken_template="SELECT CAST(COUNT(CASE WHEN {filter} THEN 1 END) AS REAL) * 100 / COUNT(CASE WHEN {wrong_denom_filter} THEN 1 END) FROM {table}",
        corrected_template="SELECT CAST(COUNT(CASE WHEN {filter} THEN 1 END) AS REAL) * 100 / COUNT(*) FROM {table}",
    ),
    # Table side errors  
    ContrastivePattern(
        pattern_id="wrong_table_side",
        name="Wrong Table Side in JOIN",
        failure_type=FailureType.WRONG_TABLE_SIDE_ERROR,
        target_failure_mode=TargetFailureMode.WRONG_TABLE_OWNERSHIP,
        description="References column from wrong side of JOIN",
        broken_template="SELECT {wrong_table}.{column} FROM {table1} JOIN {wrong_table} ON {join_condition}",
        corrected_template="SELECT {correct_table}.{column} FROM {table1} JOIN {correct_table} ON {join_condition}",
    ),
    # Table family confusion
    ContrastivePattern(
        pattern_id="table_family_confusion",
        name="Table Family Confusion",
        failure_type=FailureType.TABLE_FAMILY_CONFUSION,
        target_failure_mode=TargetFailureMode.PHASE_TABLE_CONFUSION,
        description="Uses wrong table from a related family (e.g., patients vs enrollments)",
        broken_template="SELECT {field} FROM {wrong_table} WHERE {condition}",
        corrected_template="SELECT {field} FROM {correct_table} WHERE {condition}",
    ),
    # Missing DISTINCT
    ContrastivePattern(
        pattern_id="missing_distinct",
        name="Missing DISTINCT",
        failure_type=FailureType.MISSING_DISTINCT,
        target_failure_mode=TargetFailureMode.DISTINCT_SHAPE_ERROR,
        description="Returns duplicates when unique values are needed",
        broken_template="SELECT {field} FROM {table} JOIN {join_table} ON {join_condition}",
        corrected_template="SELECT DISTINCT {field} FROM {table} JOIN {join_table} ON {join_condition}",
    ),
    # Join path errors
    ContrastivePattern(
        pattern_id="wrong_join_path",
        name="Wrong JOIN Path",
        failure_type=FailureType.JOIN_PATH_ERROR,
        target_failure_mode=TargetFailureMode.JOIN_PATH_ERROR,
        description="Joins through wrong intermediate table",
        broken_template="SELECT {field} FROM {table1} JOIN {wrong_bridge} ON {wrong_condition} JOIN {table2} ON {wrong_condition2}",
        corrected_template="SELECT {field} FROM {table1} JOIN {correct_bridge} ON {correct_condition} JOIN {table2} ON {correct_condition2}",
    ),
    # Temporal anchor errors
    ContrastivePattern(
        pattern_id="temporal_anchor",
        name="Wrong Date Anchor",
        failure_type=FailureType.TEMPORAL_ANCHOR_ERROR,
        target_failure_mode=TargetFailureMode.TEMPORAL_ANCHOR_ERROR,
        description="Uses wrong date reference (e.g., creation date vs modification date)",
        broken_template="SELECT {field} FROM {table} WHERE {date_field} {operator} {wrong_date}",
        corrected_template="SELECT {field} FROM {table} WHERE {date_field} {operator} {correct_date}",
    ),
    # Alias errors
    ContrastivePattern(
        pattern_id="alias_swap",
        name="Alias Reference Swap",
        failure_type=FailureType.ALIAS_ERROR,
        target_failure_mode=TargetFailureMode.RETURN_FIELD_MISMATCH,
        description="References wrong alias in complex query",
        broken_template="SELECT {wrong_alias}.{field} FROM {table1} AS t1 JOIN {table2} AS t2 ON {condition}",
        corrected_template="SELECT {correct_alias}.{field} FROM {table1} AS t1 JOIN {table2} AS t2 ON {condition}",
    ),
]


@dataclass
class ContrastiveStats:
    """Statistics for contrastive generation."""
    total_generated: int = 0
    by_pattern: Dict[str, int] = field(default_factory=dict)
    by_failure_type: Dict[str, int] = field(default_factory=dict)
    by_db: Dict[str, int] = field(default_factory=dict)
    clean_count: int = 0
    internal_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_generated": self.total_generated,
            "by_pattern": dict(self.by_pattern),
            "by_failure_type": dict(self.by_failure_type),
            "by_db": dict(self.by_db),
            "clean_count": self.clean_count,
            "internal_count": self.internal_count,
        }


class ContrastiveGenerator:
    """
    Generates contrastive repair examples from predefined patterns.
    
    Contrastive examples are high-value because they directly target
    commonly confused patterns identified in failure analysis.
    """
    
    def __init__(self, config: BuilderConfig):
        self.config = config
        self.stats = ContrastiveStats()
        self._counter = 0
        self._templates: Dict[str, List[Dict[str, Any]]] = {}
    
    def load_templates_from_db(
        self,
        db_id: str,
        schema: Dict[str, Any],
    ) -> int:
        """
        Load template instantiation options from a database schema.
        
        Returns number of templates loaded.
        """
        tables = schema.get("tables", [])
        if not tables:
            return 0
        
        # Extract template options
        options: Dict[str, Any] = {
            "tables": [],
            "columns": {},
            "id_columns": {},
            "name_columns": {},
            "date_columns": {},
            "join_conditions": [],
        }
        
        for table in tables:
            table_name = table.get("name", "")
            columns = table.get("columns", [])
            
            options["tables"].append(table_name)
            options["columns"][table_name] = [c.get("name", "") for c in columns]
            
            # Identify column types
            for col in columns:
                col_name = col.get("name", "").lower()
                col_type = col.get("type", "").lower()
                
                if "id" in col_name or col_name.endswith("_id"):
                    if table_name not in options["id_columns"]:
                        options["id_columns"][table_name] = []
                    options["id_columns"][table_name].append(col.get("name", ""))
                
                if "name" in col_name or "title" in col_name:
                    if table_name not in options["name_columns"]:
                        options["name_columns"][table_name] = []
                    options["name_columns"][table_name].append(col.get("name", ""))
                
                if "date" in col_name or "time" in col_name or col_type in ("date", "datetime", "timestamp"):
                    if table_name not in options["date_columns"]:
                        options["date_columns"][table_name] = []
                    options["date_columns"][table_name].append(col.get("name", ""))
        
        self._templates[db_id] = [options]
        return 1
    
    def generate_one(
        self,
        pattern: ContrastivePattern,
        db_id: str,
        schema_context: str,
        question: str,
        placeholders: Dict[str, str],
        clean_source: bool = True,
    ) -> Optional[RepairExample]:
        """
        Generate a single contrastive example.
        
        Args:
            pattern: Pattern definition
            db_id: Database ID
            schema_context: Schema text
            question: Question text
            placeholders: Values for template placeholders
            clean_source: Whether source is benchmark-clean
            
        Returns:
            RepairExample or None if generation fails
        """
        try:
            # Instantiate templates
            broken_sql = pattern.broken_template.format(**placeholders)
            corrected_sql = pattern.corrected_template.format(**placeholders)
        except KeyError:
            return None
        
        # Check they're different
        if broken_sql == corrected_sql:
            return None
        
        # Create example
        self._counter += 1
        example_id = f"contrastive_{pattern.pattern_id}_{db_id}_{self._counter}"
        
        metadata = ExampleMetadata(
            example_id=example_id,
            split="train",  # Will be assigned in splitter
            source_type=SourceType.contrastive,
            source_run=None,
            db_family=db_id,
            schema_context_type="compact_relevant",
            schema_tables_kept=[],
            schema_columns_kept=[],
            relations=[],
            notes=f"Contrastive pattern: {pattern.name}",
            failure_type=pattern.failure_type,
            target_failure_mode=pattern.target_failure_mode,
            exec_failed_originally=False,
            error_message=None,
            contamination_source=ContaminationSource.non_benchmark if clean_source else ContaminationSource.bird_dev_direct,
            benchmark_clean=clean_source,
            pool=Pool.clean if clean_source else Pool.internal,
            difficulty="moderate",
            sampling_weight=1.3,  # Contrastive examples get higher weight
            corrected_sql_source="contrastive_pattern",
            reference_sql_source="contrastive_pattern",
            verification_method="pattern_match",
            verification_passed=True,
            original_question_id=None,
            original_example_id=None,
            corruption_transform=None,
            compact_schema_stats=None,
            subagent_used=False,
            subagent_candidate_sql=None,
            subagent_accept_reason=None,
            subagent_reject_reason=None,
            subagent_retries=0,
        )
        
        example = RepairExample(
            schema_context=schema_context,
            question=question,
            hints="",
            broken_sql=broken_sql,
            corrected_sql=corrected_sql,
            error_message=None,
            failure_type_str=pattern.failure_type.value,
            metadata=metadata,
        )
        
        # Update stats
        self.stats.total_generated += 1
        self.stats.by_pattern[pattern.pattern_id] = self.stats.by_pattern.get(pattern.pattern_id, 0) + 1
        self.stats.by_failure_type[pattern.failure_type.value] = self.stats.by_failure_type.get(pattern.failure_type.value, 0) + 1
        self.stats.by_db[db_id] = self.stats.by_db.get(db_id, 0) + 1
        
        if clean_source:
            self.stats.clean_count += 1
        else:
            self.stats.internal_count += 1
        
        return example
    
    def generate_from_pattern(
        self,
        pattern: ContrastivePattern,
        count: int = 10,
    ) -> Iterator[RepairExample]:
        """
        Generate multiple examples from a single pattern.
        
        Uses loaded templates to instantiate the pattern with different values.
        """
        generated = 0
        
        for db_id, template_list in self._templates.items():
            if generated >= count:
                break
            
            for options in template_list:
                if generated >= count:
                    break
                
                tables = options.get("tables", [])
                if not tables:
                    continue
                
                # Generate placeholder variations
                for _ in range(min(3, count - generated)):
                    placeholders = self._generate_placeholders(pattern, options)
                    if not placeholders:
                        continue
                    
                    # Create minimal schema context
                    schema_context = self._create_schema_context(db_id, options, placeholders)
                    question = self._create_question(pattern, placeholders)
                    
                    example = self.generate_one(
                        pattern=pattern,
                        db_id=db_id,
                        schema_context=schema_context,
                        question=question,
                        placeholders=placeholders,
                        clean_source=True,  # Synthetic contrastive is clean
                    )
                    
                    if example:
                        generated += 1
                        yield example
    
    def generate_batch(
        self,
        count: int = 100,
        patterns: Optional[List[ContrastivePattern]] = None,
    ) -> Iterator[RepairExample]:
        """
        Generate a batch of contrastive examples across patterns.
        
        Distributes count across available patterns.
        """
        if patterns is None:
            patterns = CONTRASTIVE_PATTERNS
        
        per_pattern = max(1, count // len(patterns))
        
        for pattern in patterns:
            for example in self.generate_from_pattern(pattern, per_pattern):
                yield example
    
    def _generate_placeholders(
        self,
        pattern: ContrastivePattern,
        options: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        """Generate placeholder values for a pattern from options."""
        placeholders: Dict[str, str] = {}
        tables = options.get("tables", [])
        
        if not tables:
            return None
        
        # Select random table
        table = random.choice(tables)
        columns = options.get("columns", {}).get(table, [])
        
        if not columns:
            return None
        
        placeholders["table"] = table
        placeholders["table1"] = table
        placeholders["field"] = random.choice(columns)
        placeholders["column"] = placeholders["field"]
        
        # Add second table if available
        if len(tables) > 1:
            other_tables = [t for t in tables if t != table]
            placeholders["table2"] = random.choice(other_tables)
            placeholders["wrong_table"] = placeholders["table2"]
            placeholders["correct_table"] = table
            placeholders["join_table"] = placeholders["table2"]
        
        # ID vs name columns
        id_cols = options.get("id_columns", {}).get(table, [])
        name_cols = options.get("name_columns", {}).get(table, [])
        
        if id_cols:
            placeholders["id_field"] = random.choice(id_cols)
        else:
            placeholders["id_field"] = columns[0] if columns else "id"
        
        if name_cols:
            placeholders["name_field"] = random.choice(name_cols)
        else:
            placeholders["name_field"] = columns[-1] if len(columns) > 1 else "name"
        
        # Return field vs rank field
        if len(columns) >= 2:
            placeholders["rank_field"] = columns[0]
            placeholders["return_field"] = columns[1]
        else:
            placeholders["rank_field"] = columns[0]
            placeholders["return_field"] = columns[0]
        
        # Count fields
        placeholders["count_field"] = placeholders["id_field"]
        placeholders["wrong_field"] = placeholders["field"]
        placeholders["correct_field"] = placeholders["id_field"]
        
        # Join conditions
        placeholders["join_condition"] = f"{table}.{placeholders['id_field']} = {placeholders.get('table2', table)}.{placeholders['id_field']}"
        placeholders["condition"] = f"{placeholders['field']} IS NOT NULL"
        
        # Misc
        placeholders["direction"] = random.choice(["ASC", "DESC"])
        placeholders["limit"] = str(random.choice([1, 3, 5, 10]))
        placeholders["operator"] = random.choice([">", "<", "=", ">=", "<="])
        
        # Filters
        placeholders["filter"] = f"{placeholders['field']} > 0"
        placeholders["wrong_denom_filter"] = f"{placeholders['field']} IS NOT NULL"
        
        # Date fields
        date_cols = options.get("date_columns", {}).get(table, [])
        if date_cols:
            placeholders["date_field"] = random.choice(date_cols)
            placeholders["wrong_date"] = "'2020-01-01'"
            placeholders["correct_date"] = "'2021-01-01'"
        else:
            placeholders["date_field"] = "created_date"
            placeholders["wrong_date"] = "'2020-01-01'"
            placeholders["correct_date"] = "'2021-01-01'"
        
        # Aliases
        placeholders["wrong_alias"] = "t1"
        placeholders["correct_alias"] = "t2"
        
        # Bridge tables
        placeholders["wrong_bridge"] = placeholders.get("table2", table)
        placeholders["correct_bridge"] = table
        placeholders["wrong_condition"] = placeholders["join_condition"]
        placeholders["correct_condition"] = placeholders["join_condition"]
        placeholders["wrong_condition2"] = placeholders["join_condition"]
        placeholders["correct_condition2"] = placeholders["join_condition"]
        
        return placeholders
    
    def _create_schema_context(
        self,
        db_id: str,
        options: Dict[str, Any],
        placeholders: Dict[str, str],
    ) -> str:
        """Create a minimal schema context for the example."""
        lines = [f"Database: {db_id}", ""]
        
        table = placeholders.get("table", "")
        columns = options.get("columns", {}).get(table, [])
        
        if table and columns:
            lines.append(f"Table: {table}")
            lines.append(f"Columns: {', '.join(columns[:8])}")  # Limit columns
        
        table2 = placeholders.get("table2")
        if table2:
            columns2 = options.get("columns", {}).get(table2, [])
            if columns2:
                lines.append("")
                lines.append(f"Table: {table2}")
                lines.append(f"Columns: {', '.join(columns2[:8])}")
        
        return "\n".join(lines)
    
    def _create_question(
        self,
        pattern: ContrastivePattern,
        placeholders: Dict[str, str],
    ) -> str:
        """Create a question for the example."""
        # Generate question based on pattern type
        templates = {
            "rank_return_wrong_field": "Find the {return_field} for records with the highest {rank_field}.",
            "id_vs_name": "What is the {name_field} where {condition}?",
            "count_vs_count_distinct": "Count the unique {count_field} values where {condition}.",
            "count_distinct_wrong_field": "How many distinct {correct_field} values are there?",
            "percentage_wrong_denominator": "What percentage of records have {filter}?",
            "wrong_table_side": "Get the {column} from {correct_table}.",
            "table_family_confusion": "Find the {field} from {correct_table}.",
            "missing_distinct": "List the unique {field} values.",
            "wrong_join_path": "Find {field} through the correct relationship.",
            "temporal_anchor": "Find records where {date_field} is {operator} {correct_date}.",
            "alias_swap": "Get {field} from the correct table.",
        }
        
        template = templates.get(pattern.pattern_id, "Find the requested information.")
        
        try:
            return template.format(**placeholders)
        except KeyError:
            return "Find the requested information."
    
    def get_stats(self) -> ContrastiveStats:
        """Get generation statistics."""
        return self.stats
    
    def get_patterns(self) -> List[ContrastivePattern]:
        """Get available patterns."""
        return CONTRASTIVE_PATTERNS
