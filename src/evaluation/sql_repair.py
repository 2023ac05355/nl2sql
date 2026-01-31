import re


def extract_select_statement(text: str) -> str | None:
    """
    Extract the first SELECT ... FROM ... SQL statement.
    """
    match = re.search(r"(select\s+.*?\s+from\s+\w+.*?)(?:;|$)", text.lower(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def clean_sql_output(text: str) -> str:
    """
    Remove markdown fences and explanations from LLM output.
    """
    text = text.strip()

    # Remove ```sql fences
    if text.startswith("```"):
        text = text.split("```")[1]

    # Remove trailing explanations
    text = text.split("\n\n")[0]

    return text.strip()
