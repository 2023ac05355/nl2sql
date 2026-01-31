"""
Gemini-based Text-to-SQL model wrapper.

Provides a clean, deterministic interface to Google's Gemini API
for Text-to-SQL benchmarking.

This module is designed for academic evaluation:
- No chain-of-thought leakage
- Deterministic generation (temperature = 0)
- Model-agnostic prompt handling
"""

import os
from typing import Optional
import google.generativeai as genai
import dotenv

dotenv.load_dotenv()
loaded_api_key = os.getenv("GEMINI_API_KEY")

class GeminiText2SQL:
    """
    Wrapper for Google Gemini API to generate SQL queries from natural language.
    """

    def __init__(
        self,
        model_name: str = "models/gemini-2.5-flash",
        api_key: Optional[str] = os.getenv("GEMINI_API_KEY"),
    ):
        """
        Initialize the Gemini Text-to-SQL model.

        Args:
            model_name: Fully-qualified Gemini model ID
                        (e.g., 'models/gemini-1.5-flash')
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY
                     environment variable.

        Raises:
            ValueError: If API key is missing.
        """
        if api_key is None:
            api_key = loaded_api_key

        if not api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Set GEMINI_API_KEY environment variable or pass api_key explicitly."
            )

        # Configure Gemini client
        genai.configure(api_key=api_key)

        # Initialize model
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def generate_sql(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        """
        Generate SQL query from a Text-to-SQL prompt.

        Args:
            prompt: Full Text-to-SQL prompt (instruction + schema + question)
            temperature: Sampling temperature (0.0 = deterministic, recommended)
            max_tokens: Maximum number of output tokens

        Returns:
            Generated SQL query as a string

        Raises:
            RuntimeError: If Gemini API call fails or returns empty output
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )

            # Gemini sometimes returns None or empty text
            if response is None or not hasattr(response, "text"):
                raise RuntimeError("Gemini returned an empty response.")

            return response.text.strip()

        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}") from e
