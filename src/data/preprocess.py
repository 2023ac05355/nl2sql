"""
Preprocessing for Text-to-SQL experiments.

This module controls what information is exposed to the model.
"""

from typing import Tuple
from .load_spider import SpiderExample, TableSchema


def linearize_schema(schema: TableSchema) -> str:
    """
    Linearize schema with tables and columns (Spider-style).
    """
    table_map = {}
    for table_idx, col_name in schema.column_names_original:
        if table_idx == -1:
            continue
        table_name = schema.table_names_original[table_idx]
        table_map.setdefault(table_name, []).append(col_name)

    parts = []
    for table, cols in table_map.items():
        cols_str = ", ".join(cols)
        parts.append(f"{table}({cols_str})")

    return " | ".join(parts)



def build_input(example: SpiderExample, schema: TableSchema) -> str:
    """
    Build model input text from question and schema.
    """
    schema_text = linearize_schema(schema)
    return f"Question: {example.question}\nSchema: {schema_text}"


def build_target(example: SpiderExample) -> str:
    """
    Return target SQL query.
    """
    return example.query


def build_gemini_zeroshot_input(example, schema):
    """
    Strict zero-shot Text-to-SQL prompt for Gemini.
    """
    schema_text = linearize_schema(schema)

    prompt = f"""
        You are a Text-to-SQL system.

        Task:
        Given the database schema and the question, write a VALID SQLite SQL query.

        Rules:
        - Output ONLY the SQL query
        - Do NOT include explanations
        - Do NOT include markdown
        - Do NOT include comments
        - Do NOT include multiple queries
        - The query MUST be executable in SQLite

        Schema:
        {schema_text}

        Question:
        {example.question}

        SQL:
        """
    return prompt.strip()


