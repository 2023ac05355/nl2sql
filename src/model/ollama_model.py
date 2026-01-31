import subprocess
from ..model.llm_interface import LLMText2SQL


class OllamaText2SQL(LLMText2SQL):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_sql(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model_name],
            input=prompt,
            text=True,
            capture_output=True
        )
        return result.stdout.strip()
