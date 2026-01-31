"""
Minimal inference test for Text-to-SQL model.
"""

from ..model.base_model import BaseText2SQLModel
from ..data.load_spider import SpiderDataLoader
from ..data.preprocess import build_input


def run_inference():
    loader = SpiderDataLoader("spider_data")
    examples = loader.load_train()
    schemas = loader.load_schemas()

    example = examples[0]
    schema = schemas[example.db_id]

    model_input = build_input(example, schema)

    model = BaseText2SQLModel(model_name="t5-small")
    predicted_sql = model.generate_sql(model_input)

    print("📝 QUESTION:")
    print(example.question)

    print("\n🧩 MODEL INPUT:")
    print(model_input)

    print("\n🤖 PREDICTED SQL:")
    print(predicted_sql)

    print("\n🎯 GOLD SQL:")
    print(example.query)


if __name__ == "__main__":
    run_inference()
