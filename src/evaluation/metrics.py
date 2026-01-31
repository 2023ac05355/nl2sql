"""
Evaluation metrics for Text-to-SQL benchmarking.
"""

import re
import sqlite3



def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison:
    - lowercase
    - remove extra whitespace
    - remove trailing semicolons
    """
    sql = sql.lower().strip()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.rstrip(";")
    return sql


def exact_match(pred_sql: str, gold_sql: str) -> bool:
    """
    Exact string match after normalization.
    """
    return normalize_sql(pred_sql) == normalize_sql(gold_sql)




def execute_sql(db_path: str, sql: str):
    """
    Execute SQL on SQLite database and return results.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    conn.close()
    return result


def execution_match(pred_sql: str, gold_sql: str, db_path: str) -> bool:
    """
    Compare execution results of predicted and gold SQL.
    """
    try:
        pred_result = execute_sql(db_path, pred_sql)
        gold_result = execute_sql(db_path, gold_sql)
        return pred_result == gold_result
    except Exception:
        # invalid SQL, execution error
        return False
