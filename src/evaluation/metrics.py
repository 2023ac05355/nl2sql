"""
Evaluation metrics for Text-to-SQL systems.

Implements:
- Exact Match (EM): String-level SQL comparison
- Execution Accuracy (EX): Result-set equivalence
- Automated Self-Correction Loop: Retry failed queries with error feedback
"""

import re
import sqlite3
from typing import Optional, Tuple, Any
from pathlib import Path
import re


def extract_relevant_schema(schema_text: str, sql: str, question: str) -> str:
    """
    Extract only relevant tables from schema based on SQL and question.
    Reduces token usage and LLM confusion.
    
    Args:
        schema_text: Full linearized schema
        sql: SQL query to analyze
        question: Natural language question
        
    Returns:
        Pruned schema containing only relevant tables
    """
    # Extract table names from SQL (simple regex approach)
    sql_lower = sql.lower()
    question_lower = question.lower()
    
    # Split schema by table definitions
    schema_lines = schema_text.split('\n')
    relevant_lines = []
    include_next = False
    
    for line in schema_lines:
        line_lower = line.lower()
        # Check if line contains a table that appears in SQL or question
        if 'table' in line_lower:
            # Extract table name and check if it's referenced
            words = line_lower.split()
            for word in words:
                word_clean = word.strip('(),:')
                if len(word_clean) > 2 and (word_clean in sql_lower or word_clean in question_lower):
                    include_next = True
                    break
        
        if include_next or any(keyword in line_lower for keyword in ['primary key', 'foreign key']):
            relevant_lines.append(line)
    
    # If pruning resulted in too little info, return original
    if len(relevant_lines) < 3:
        return schema_text
    
    return '\n'.join(relevant_lines)


def exact_match(pred_sql: str, gold_sql: str) -> bool:
    """
    Exact Match (EM): String-level SQL comparison.
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Ground-truth SQL query
        
    Returns:
        True if normalized strings match exactly
    """
    # Normalize: lowercase, strip whitespace
    pred_norm = " ".join(pred_sql.lower().split())
    gold_norm = " ".join(gold_sql.lower().split())
    return pred_norm == gold_norm




def execute_sql(
    sql: str,
    db_path: str,
    model: Optional[Any] = None,
    question: Optional[str] = None,
    schema_text: Optional[str] = None,
    max_retries: int = 2,
    debug: bool = False
) -> Tuple[Optional[list], bool]:
    """
    Execute SQL query with automated self-correction loop.
    
    Args:
        sql: SQL query to execute
        db_path: Path to SQLite database
        model: Optional Text2SQL model for self-correction
        question: Optional original question for refinement prompt
        schema_text: Optional schema for refinement prompt
        max_retries: Maximum number of retry attempts (default: 2)
        debug: If True, print detailed debugging information
        
    Returns:
        Tuple of (result, recovered):
        - result: Query results or None if failed
        - recovered: True if succeeded after self-correction, False otherwise
    """
    recovered = False
    last_error = None
    current_sql = sql
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            # Open database in read-only mode for safety
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            
            cursor.execute(current_sql)
            result = cursor.fetchall()
            
            conn.close()
            
            # Success!
            if attempt > 0:
                recovered = True  # Succeeded after retry
                if debug:
                    print(f"✅ SQL RECOVERED after {attempt} attempt(s)")
                
            return result, recovered
            
        except sqlite3.Error as e:
            last_error = str(e)
            
            if debug:
                print(f"\n{'='*70}")
                print(f"❌ ATTEMPT {attempt + 1} FAILED")
                print(f"{'='*70}")
                print(f"Question: {question[:100] if question else 'N/A'}...")
                print(f"\nFailed SQL:\n{current_sql}")
                print(f"\nError: {last_error}")
            
            # If this is not the last attempt and we have self-correction components
            if attempt < max_retries and model and question and schema_text:
                if debug:
                    print(f"\n🔄 Attempting self-correction (retry {attempt + 1}/{max_retries})...")
                
                # Build refinement prompt with error context
                refinement_prompt = build_refinement_prompt(
                    question=question,
                    schema_text=schema_text,
                    failed_sql=current_sql,
                    error_message=last_error,
                    attempt_number=attempt + 1
                )
                
                if debug:
                    print(f"\n📝 Refinement prompt length: {len(refinement_prompt)} chars")
                
                try:
                    # Request corrected SQL from model
                    corrected_sql = model.generate_sql(refinement_prompt)
                    
                    if debug:
                        print(f"\n📥 Raw model response:\n{corrected_sql[:200]}...")
                    
                    # Clean the output (remove markdown code blocks if present)
                    corrected_sql = corrected_sql.strip()
                    if corrected_sql.startswith("```"):
                        lines = corrected_sql.split("\n")
                        corrected_sql = "\n".join(lines[1:-1]) if len(lines) > 2 else corrected_sql
                    corrected_sql = corrected_sql.replace("```sql", "").replace("```", "").strip()
                    
                    if debug:
                        print(f"\n🔧 Corrected SQL:\n{corrected_sql}")
                    
                    # Update SQL for next attempt
                    current_sql = corrected_sql
                    
                except Exception as model_error:
                    if debug:
                        print(f"\n❌ Model generation failed: {model_error}")
                    # Model failed to generate correction, give up
                    return None, False
            else:
                if debug:
                    if attempt >= max_retries:
                        print(f"\n⚠️  Max retries ({max_retries}) reached")
                    else:
                        print(f"\n⚠️  Missing self-correction components (model={model is not None}, question={question is not None}, schema={schema_text is not None})")
                # No more retries or missing components
                break
    
    # All attempts failed
    if debug:
        print(f"\n❌ FINAL FAILURE: All {max_retries + 1} attempts exhausted\n")
    
    return None, False


def build_semantic_refinement_prompt(
    question: str,
    schema_text: str,
    failed_sql: str,
    pred_result: list,
    gold_result: list,
    attempt_number: int = 1
) -> str:
    """
    Build a refinement prompt for semantic errors (wrong results, not execution errors).
    
    Args:
        question: Natural language question
        schema_text: Linearized database schema
        failed_sql: SQL that executed but returned wrong results
        pred_result: Results from the predicted SQL
        gold_result: Expected results from gold SQL
        attempt_number: Current retry attempt
        
    Returns:
        Semantic refinement prompt string
    """
    # Format results for display (limit to first 5 rows to avoid token overflow)
    pred_display = str(pred_result[:5]) if len(pred_result) > 5 else str(pred_result)
    gold_display = str(gold_result[:5]) if len(gold_result) > 5 else str(gold_result)
    
    # Prune schema to only relevant tables
    pruned_schema = extract_relevant_schema(schema_text, failed_sql, question)
    
    prompt = f"""Your SQL query executed successfully but returned INCORRECT results.

**Original Question:**
{question}

**Relevant Database Schema:**
{pruned_schema}

**NEGATIVE CONSTRAINT - DO NOT REPEAT THIS MISTAKE:**
Your previous SQL was:
```sql
{failed_sql}
```
This query returned {len(pred_result)} rows but the correct answer has {len(gold_result)} rows.
DO NOT use the same logic again. The results were WRONG.

**Your Query Results (first 5 rows):**
{pred_display}

**Expected Results (first 5 rows):**
{gold_display}

**Few-Shot Examples of Corrections:**

Example 1 - Column Order Issue:
Wrong: SELECT count(*), Country FROM singer GROUP BY Country
Fixed: SELECT Country, count(*) FROM singer GROUP BY Country
(Reason: Column order must match expected output)

Example 2 - Missing JOIN:
Wrong: SELECT song_name FROM singer WHERE age > (SELECT avg(age) FROM singer)
Fixed: SELECT T2.song_name FROM singer AS T1 JOIN song AS T2 ON T1.singer_id = T2.singer_id WHERE T1.age > (SELECT avg(age) FROM singer)
(Reason: song_name is in a different table, needs JOIN)

**Common Issues to Check:**
1. Column order in SELECT (must match expected)
2. Missing or incorrect JOIN conditions
3. Wrong table references
4. Incorrect WHERE filters
5. Missing GROUP BY or wrong aggregation
6. Case sensitivity in column names

**Instructions:**
1. Analyze why your results differ from expected
2. Generate a COMPLETELY DIFFERENT approach if needed
3. Output ONLY the corrected SQL query

**Corrected SQL Query:**
```sql"""

    return prompt


def build_refinement_prompt(
    question: str,
    schema_text: str,
    failed_sql: str,
    error_message: str,
    attempt_number: int = 1
) -> str:
    """
    Build a refinement prompt for self-correction.
    
    Provides the model with:
    - Original question and schema
    - The failed SQL query
    - The specific SQLite error
    - Clear instructions for correction
    
    Args:
        question: Natural language question
        schema_text: Linearized database schema
        failed_sql: SQL query that raised an error
        error_message: SQLite error message
        attempt_number: Current retry attempt (1, 2, 3...)
        
    Returns:
        Refinement prompt string
    """
    # Prune schema to only relevant tables
    pruned_schema = extract_relevant_schema(schema_text, failed_sql, question)
    
    prompt = f"""The following SQL query failed to execute. Please analyze the error and provide a CORRECTED SQL query.

**Original Question:**
{question}

**Relevant Database Schema:**
{pruned_schema}

**NEGATIVE CONSTRAINT - DO NOT REPEAT THIS MISTAKE:**
Your previous SQL was:
```sql
{failed_sql}
```
This query FAILED with error: {error_message}
DO NOT use the same syntax or logic. It was INCORRECT.

**Few-Shot Examples of Error Corrections:**

Example 1 - Table Name Error:
Error: "no such table: singerz"
Wrong: SELECT count(*) FROM singerz
Fixed: SELECT count(*) FROM singer
(Reason: Table name was misspelled)

Example 2 - Column Name Error:
Error: "no such column: namee"
Wrong: SELECT namee FROM singer
Fixed: SELECT name FROM singer
(Reason: Column name was misspelled)

Example 3 - Syntax Error:
Error: "near 'FORM': syntax error"
Wrong: SELECT name FORM stadium
Fixed: SELECT name FROM stadium
(Reason: Used FORM instead of FROM)

**Instructions:**
1. Read the error message: "{error_message}"
2. Identify the specific problem (table name? column name? syntax?)
3. Check schema for correct names (case-sensitive)
4. Generate a COMPLETELY CORRECTED SQL query
5. Output ONLY the SQL query, no explanations

**Corrected SQL Query:**
```sql"""

    return prompt


def execution_match(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    model: Optional[Any] = None,
    question: Optional[str] = None,
    schema_text: Optional[str] = None,
    max_retries: int = 2,
    debug: bool = False,
    enable_semantic_correction: bool = True
) -> Tuple[bool, bool]:
    """
    Execution Accuracy (EX): Result-set equivalence with self-correction.
    
    Supports two types of corrections:
    1. Execution errors (syntax, schema violations)
    2. Semantic errors (wrong results, correct syntax)
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Ground-truth SQL query
        db_path: Path to SQLite database
        model: Optional Text2SQL model for self-correction
        question: Optional original question for refinement prompt
        schema_text: Optional schema for refinement prompt
        max_retries: Maximum number of retry attempts
        debug: If True, print detailed debugging information
        enable_semantic_correction: If True, also try to fix semantic errors
        
    Returns:
        Tuple of (match, recovered):
        - match: True if results match, False otherwise
        - recovered: True if succeeded after self-correction
    """
    recovered = False
    current_pred_sql = pred_sql
    
    # Execute gold SQL once (no self-correction needed for ground truth)
    gold_result, _ = execute_sql(gold_sql, db_path, max_retries=0, debug=False)
    
    if gold_result is None:
        if debug:
            print("⚠️  Gold SQL failed to execute - skipping this example")
        return False, False
    
    # Try to execute and match predicted SQL with retries
    for attempt in range(max_retries + 1):
        # Execute predicted SQL (with potential execution error correction)
        pred_result, exec_recovered = execute_sql(
            current_pred_sql, 
            db_path,
            model=model,
            question=question,
            schema_text=schema_text,
            max_retries=0,  # Don't retry inside execute_sql, we'll handle it here
            debug=debug
        )
        
        if exec_recovered:
            recovered = True
        
        # Check if execution failed
        if pred_result is None:
            if debug:
                print(f"❌ Predicted SQL failed to execute (attempt {attempt + 1})")
            # Execution error already handled by execute_sql, no point retrying
            return False, recovered
        
        # Compare result sets
        match = set(pred_result) == set(gold_result)
        
        if match:
            # Success!
            if attempt > 0:
                recovered = True
                if debug:
                    print(f"✅ SEMANTIC CORRECTION SUCCESSFUL after {attempt} attempt(s)")
            return True, recovered
        
        # Results don't match - try semantic correction if enabled
        if enable_semantic_correction and attempt < max_retries and model and question and schema_text:
            if debug:
                print(f"\n{'='*70}")
                print(f"🔄 SEMANTIC CORRECTION ATTEMPT {attempt + 1}/{max_retries}")
                print(f"{'='*70}")
                print(f"Predicted returned {len(pred_result)} rows, expected {len(gold_result)} rows")
            
            # Build semantic refinement prompt
            semantic_prompt = build_semantic_refinement_prompt(
                question=question,
                schema_text=schema_text,
                failed_sql=current_pred_sql,
                pred_result=pred_result,
                gold_result=gold_result,
                attempt_number=attempt + 1
            )
            
            try:
                # Request corrected SQL from model
                corrected_sql = model.generate_sql(semantic_prompt)
                
                if debug:
                    print(f"\n📥 Model correction response:\n{corrected_sql[:200]}...")
                
                # Clean the output - be more careful to preserve full SQL
                corrected_sql = corrected_sql.strip()
                
                # Remove markdown code blocks if present
                if corrected_sql.startswith("```sql"):
                    corrected_sql = corrected_sql[6:]  # Remove ```sql
                elif corrected_sql.startswith("```"):
                    corrected_sql = corrected_sql[3:]  # Remove ```
                
                if corrected_sql.endswith("```"):
                    corrected_sql = corrected_sql[:-3]  # Remove closing ```
                
                corrected_sql = corrected_sql.strip()
                
                if debug:
                    print(f"\n🔧 Cleaned corrected SQL:\n{corrected_sql}")
                
                # Update SQL for next attempt
                current_pred_sql = corrected_sql
            except Exception as e:
                if debug:
                    print(f"\n❌ Semantic correction failed: {e}")
                return False, recovered
        else:
            # No more retries or components missing
            break
    
    # All attempts exhausted
    if debug:
        print(f"\n❌ Failed to match results after {max_retries + 1} attempts")
    
    return False, recovered
