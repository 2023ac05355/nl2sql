
"""
Unified interface for LLM-based Text-to-SQL generation.
"""

class LLMText2SQL:
    def generate_sql(self, prompt: str) -> str:
        raise NotImplementedError
