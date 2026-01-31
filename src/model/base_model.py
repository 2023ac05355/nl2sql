"""
Base Text-to-SQL model wrapper.

This module provides a thin abstraction over a pretrained
sequence-to-sequence model for SQL generation.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class BaseText2SQLModel:
    """
    Inference-only wrapper for a Text-to-SQL model.
    """

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate_sql(self, input_text: str, max_length: int = 128) -> str:
        """
        Generate SQL from natural language input.

        Args:
            input_text: Preprocessed model input
            max_length: Maximum output token length

        Returns:
            Generated SQL query as string
        """
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length
            )

        sql = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return sql
