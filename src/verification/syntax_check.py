"""
Syntactic Verification Layer for Text-to-SQL System

This module provides lightweight syntactic validation of SQL queries
before execution. It is designed for an academic NL2SQL system to
implement early rejection of malformed queries.

Design Philosophy:
    "The goal is not exhaustive correctness but controlled risk reduction
    with minimal overhead."
    
    This layer validates SQL structure, not semantics or authorization.

Approach:
    Uses SQLite's execution engine as a pragmatic proxy for syntactic validity.
    This avoids the complexity of implementing a full SQL grammar parser.
    
Justification:
    - SQL parsing requires complex grammar rules (ANTLR, AST, etc.)
    - SQLite already validates syntax during execution
    - Execution-based validation is sufficient for this academic project
    - False negatives (e.g., wrong column names) are acceptable by design
    
Note:
    This layer intentionally treats schema errors (invalid tables/columns)
    as syntax failures, as both prevent successful execution in SQLite.
    This is a deliberate design choice for simplicity.
"""

import sqlite3
from pathlib import Path


def syntax_check(sql: str, db_path: str) -> bool:
    """
    Verify syntactic validity of a SQL query using SQLite.
    
    This function attempts to execute the SQL statement on a SQLite
    database to determine if it is syntactically valid. Any execution
    errors (syntax errors, invalid table/column names, etc.) result
    in False.
    
    Implementation Strategy:
        - Connects to the target database
        - Attempts to execute the SQL
        - Returns True if execution succeeds or would succeed
        - Returns False for any errors (syntax, schema mismatch, etc.)
    
    Args:
        sql: SQL query string to validate
        db_path: Absolute path to SQLite database file
        
    Returns:
        bool: True if SQL is syntactically valid, False otherwise
        
    Examples:
        >>> syntax_check("SELECT * FROM users", "test.db")
        True
        
        >>> syntax_check("SELECTT * FROM users", "test.db")
        False
        
        >>> syntax_check("SELECT * FROM nonexistent_table", "test.db")
        False
    
    Academic Justification:
        This function does NOT modify the database (read-only connection).
        It is safe to use on production databases.
        
        False negatives are acceptable by design - this layer aims for
        controlled risk reduction, not exhaustive validation.
        
        Invalid table/column names are treated as syntax failures because
        they prevent successful execution, which is the practical concern.
    """
    # Input validation: empty SQL is invalid
    if not sql or not sql.strip():
        return False
    
    # Database must exist
    if not Path(db_path).exists():
        return False
    
    try:
        # Connect in READ-ONLY mode (critical for safety)
        # URI mode required for read-only flag
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        # Attempt execution: syntax errors will raise sqlite3.Error
        cursor.execute(sql)
        
        # Fetch all results to ensure query completes successfully
        # Some errors (e.g., type mismatches) only appear during fetching
        cursor.fetchall()
        
        # Clean up connection
        conn.close()
        
        # Success: SQL is syntactically valid and executable
        return True
        
    except sqlite3.Error:
        # Any SQLite error is treated as syntax failure
        # Includes:
        #   - Syntax errors (sqlite3.OperationalError)
        #   - Invalid table/column names (schema errors)
        #   - Type mismatches
        #   - Constraint violations
        # No logging or re-raising by design
        return False
        
    except Exception:
        # Catch any other unexpected errors
        # Examples: connection issues, file permissions, encoding errors
        # Suppress all exceptions (no logging, no re-raising)
        return False


# Backward compatibility: original simple implementation
def syntax_check_simple(sql: str, db_path: str) -> bool:
    """
    Simplified syntax check (original implementation).
    
    Kept for backward compatibility with existing code.
    Consider using syntax_check() for production use.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.close()
        return True
    except Exception:
        return False


__all__ = ['syntax_check', 'syntax_check_simple']

