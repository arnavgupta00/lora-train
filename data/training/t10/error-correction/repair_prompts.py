#!/usr/bin/env python3
"""
Repair Prompts Module

Defines repair prompt templates for each failure type.
All prompts enforce:
- Minimal edit to original SQL
- Use only provided identifiers
- Do not invent columns/tables
- Copy exact names with proper quoting
- Output only SQL
"""

from typing import Any, Dict, Optional

from extract_relevant_schema import RelevantSchemaBlock


# =============================================================================
# System Prompt
# =============================================================================

REPAIR_SYSTEM_PROMPT = """You are an expert SQL repair assistant. Fix SQL errors using only the provided schema.
Rules:
- Make minimal edits to the original SQL
- Use only tables and columns from the provided schema
- Copy identifier names exactly (including backticks for special characters)
- Do not add new tables unless absolutely necessary
- Do not change join structure unless it is necessary to fix the execution error
- Output only the corrected SQL, nothing else"""


# =============================================================================
# Prompt Templates
# =============================================================================

PROMPT_TEMPLATES = {
    "fix_exact_identifier_error": """{system_tag}
The SQL below fails with: {error_message}

Schema:
{schema_block}

Question: {question}
Hints: {hints}

Original SQL (with error):
{predicted_sql}

The column/table `{failed_identifier}` does not exist.
{suggestion_line}

Fix the SQL by replacing ONLY the incorrect identifier. Do not add new tables unless absolutely necessary.
Make the minimal edit possible.
Output only the corrected SQL:""",

    "fix_wrong_table_side_error": """{system_tag}
The SQL below fails with: {error_message}

Schema:
{schema_block}

Question: {question}
Hints: {hints}

Original SQL (with error):
{predicted_sql}

The column `{failed_identifier}` exists but is referenced from the wrong table/alias.
It belongs to table `{correct_table}`, not `{wrong_alias}`.

Fix ONLY the table alias/reference. Do not change anything else.
Output only the corrected SQL:""",

    "fix_alias_error": """{system_tag}
The SQL below fails because of incorrect table alias usage: {error_message}

Schema:
{schema_block}

Question: {question}
Hints: {hints}

Original SQL (with error):
{predicted_sql}

The alias `{wrong_alias}` is used but does not correspond to the table containing column `{failed_identifier}`.
{suggestion_line}

Fix ONLY the alias reference. Do not restructure the query.
Output only the corrected SQL:""",

    "fix_degenerate_sql": """{system_tag}
The SQL below is incomplete or malformed: {error_message}

Schema:
{schema_block}

Question: {question}
Hints: {hints}

Broken SQL:
{predicted_sql}

Complete the SQL query properly. Keep the structure similar to the original.
Do not add new tables unless absolutely necessary.
Output only the corrected SQL:""",

    "fix_syntax_error": """{system_tag}
The SQL below has a syntax error: {error_message}

Schema:
{schema_block}

Question: {question}
Hints: {hints}

Original SQL (with error):
{predicted_sql}

{notes_section}

Fix the syntax error. If a column name contains spaces or special characters, wrap it in backticks.
Make the minimal edit possible.
Output only the corrected SQL:""",

    "fix_aggregate_error": """{system_tag}
The SQL below has an aggregate function error: {error_message}

Schema:
{schema_block}

Question: {question}
Hints: {hints}

Original SQL (with error):
{predicted_sql}

The error is likely due to:
- Using aggregate (COUNT, SUM, etc.) in WHERE instead of HAVING
- Missing GROUP BY clause
- Aggregate in wrong position

Fix the aggregate usage. Make the minimal edit possible.
Output only the corrected SQL:""",

    "fix_generic_exec_error": """{system_tag}
The SQL below fails with: {error_message}

Schema:
{schema_block}

Relations:
{relations}

Question: {question}
Hints: {hints}

Original SQL (with error):
{predicted_sql}

{notes_section}

Analyze the error and fix the SQL.
IMPORTANT: Make the minimal change necessary. Do not add new tables unless absolutely required.
Output only the corrected SQL:""",

    "fix_filter_value_error": """{system_tag}
The SQL has a filter/value mapping error. The WHERE or HAVING clause likely has:
- Wrong literal value
- Type mismatch (string vs number)
- Incorrect LIKE pattern

Schema:
{schema_block}

Relations:
{relations}

Question: {question}
Hints: {hints}

Original SQL (with error):
{predicted_sql}

Error:
{error_message}

{notes_section}

Fix the filter condition using only values that match the column's data type.
Use the schema column types, question, and hints as reference.
Do not change the overall query structure.
Do not add new tables unless absolutely necessary.
Output only the corrected SQL.""",

    "fix_join_backbone_error": """{system_tag}
The SQL has a join structure error, likely:
- Ambiguous column reference (missing table qualifier)
- Missing or incorrect JOIN ON condition
- Cartesian product

Schema:
{schema_block}

Relations:
{relations}

Question: {question}
Hints: {hints}

Original SQL (with error):
{predicted_sql}

Error:
{error_message}

{notes_section}

Fix the join structure:
- Add table qualifiers to ambiguous columns
- Ensure all JOINs have proper ON conditions using the relations provided
- Preserve the original join structure unless a join change is necessary to fix the execution error
- Do not add new tables unless absolutely necessary
Output only the corrected SQL.""",
}


# =============================================================================
# Prompt Builder
# =============================================================================

class RepairPromptBuilder:
    """Builds repair prompts from classification and schema."""
    
    def __init__(self, enable_thinking: bool = True):
        """
        Args:
            enable_thinking: If True, allow model to think internally during repair
        """
        self.enable_thinking = enable_thinking
    
    def build_prompt(
        self,
        failure_type: str,
        schema_block: RelevantSchemaBlock,
        question: str,
        hints: str,
        predicted_sql: str,
        error_message: str,
        failed_identifier: Optional[str] = None,
        suggested_fix: Optional[str] = None,
        wrong_alias: Optional[str] = None,
        correct_table: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Build a repair prompt.
        
        Returns:
            Dict with 'system' and 'user' keys
        """
        # Select template based on failure type
        template_key = self._get_template_key(failure_type)
        template = PROMPT_TEMPLATES.get(template_key, PROMPT_TEMPLATES["fix_generic_exec_error"])
        
        # Build system tag (thinking mode)
        system_tag = "/think" if self.enable_thinking else "/no_think"
        
        # Build suggestion line
        suggestion_line = ""
        if suggested_fix:
            suggestion_line = f"Looking at the schema, the correct identifier is likely `{suggested_fix}`."
        
        # Build notes section
        notes_section = ""
        if schema_block.notes:
            notes_section = "Notes:\n" + schema_block.format_notes()
        
        # Format prompt
        user_content = template.format(
            system_tag=system_tag,
            error_message=error_message,
            schema_block=schema_block.format_ddl(),
            question=question,
            hints=hints if hints else "None",
            predicted_sql=predicted_sql,
            failed_identifier=failed_identifier or "unknown",
            suggested_fix=suggested_fix or "unknown",
            suggestion_line=suggestion_line,
            wrong_alias=wrong_alias or "unknown",
            correct_table=correct_table or "unknown",
            relations=schema_block.format_relations(),
            notes_section=notes_section,
        )
        
        return {
            "system": REPAIR_SYSTEM_PROMPT,
            "user": user_content,
        }
    
    def _get_template_key(self, failure_type: str) -> str:
        """Map failure type to template key."""
        mapping = {
            "exact_identifier_error": "fix_exact_identifier_error",
            "wrong_table_side_error": "fix_wrong_table_side_error",
            "alias_error": "fix_alias_error",
            "degenerate_or_truncated_sql": "fix_degenerate_sql",
            "derived_metric_error": "fix_aggregate_error",
            "generic_exec_error": "fix_generic_exec_error",
            "filter_value_mapping_error": "fix_filter_value_error",
            "join_backbone_error": "fix_join_backbone_error",
        }
        return mapping.get(failure_type, "fix_generic_exec_error")
    
    def build_messages(
        self,
        failure_type: str,
        schema_block: RelevantSchemaBlock,
        question: str,
        hints: str,
        predicted_sql: str,
        error_message: str,
        **kwargs,
    ) -> list:
        """
        Build chat messages for repair.
        
        Returns:
            List of message dicts with 'role' and 'content'
        """
        prompt = self.build_prompt(
            failure_type=failure_type,
            schema_block=schema_block,
            question=question,
            hints=hints,
            predicted_sql=predicted_sql,
            error_message=error_message,
            **kwargs,
        )
        
        return [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]


# =============================================================================
# Second Attempt Prompts (Escalation)
# =============================================================================

ESCALATION_TEMPLATE = """{system_tag}
The previous repair attempt failed. The SQL still has an error.

Schema:
{schema_block}

Relations:
{relations}

Question: {question}
Hints: {hints}

Original SQL (with error):
{predicted_sql}

First repair attempt:
{first_repair_sql}

First repair error: {first_repair_error}

Previous error was: {original_error}

Please try a different approach to fix the SQL.
Use ONLY the columns and tables from the schema above.
Check that all column names are spelled exactly as shown in the schema.
If a column has spaces or special characters, use backticks.
Output only the corrected SQL:"""


class EscalationPromptBuilder:
    """Builds escalation prompts for second repair attempts."""
    
    def __init__(self, enable_thinking: bool = True):
        self.enable_thinking = enable_thinking
    
    def build_prompt(
        self,
        schema_block: RelevantSchemaBlock,
        question: str,
        hints: str,
        predicted_sql: str,
        original_error: str,
        first_repair_sql: str,
        first_repair_error: str,
    ) -> Dict[str, str]:
        """Build escalation prompt for second attempt."""
        system_tag = "/think" if self.enable_thinking else "/no_think"
        
        user_content = ESCALATION_TEMPLATE.format(
            system_tag=system_tag,
            schema_block=schema_block.format_ddl(),
            relations=schema_block.format_relations(),
            question=question,
            hints=hints if hints else "None",
            predicted_sql=predicted_sql,
            original_error=original_error,
            first_repair_sql=first_repair_sql,
            first_repair_error=first_repair_error,
        )
        
        return {
            "system": REPAIR_SYSTEM_PROMPT,
            "user": user_content,
        }
    
    def build_messages(
        self,
        schema_block: RelevantSchemaBlock,
        question: str,
        hints: str,
        predicted_sql: str,
        original_error: str,
        first_repair_sql: str,
        first_repair_error: str,
    ) -> list:
        """Build chat messages for escalation."""
        prompt = self.build_prompt(
            schema_block=schema_block,
            question=question,
            hints=hints,
            predicted_sql=predicted_sql,
            original_error=original_error,
            first_repair_sql=first_repair_sql,
            first_repair_error=first_repair_error,
        )
        
        return [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Repair Prompts Module")
    print("=" * 60)
    print(f"System prompt: {REPAIR_SYSTEM_PROMPT[:50]}...")
    print(f"Templates available: {list(PROMPT_TEMPLATES.keys())}")
