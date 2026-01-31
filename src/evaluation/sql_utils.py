import sqlite3


def is_executable(sql: str, db_path: str) -> bool:
    """
    Check whether SQL can be executed without error.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        cursor.fetchall()
        conn.close()
        return True
    except Exception:
        return False
