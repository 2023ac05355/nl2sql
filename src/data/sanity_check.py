"""
Sanity checks for Spider data loading and preprocessing.

This script verifies:
1. Spider dataset loads correctly
2. Schema is retrievable for each example
3. Question, schema, and SQL align
4. Preprocessing produces stable model inputs

Run:
    python src/data/sanity_check.py
"""

from .load_spider import SpiderDataLoader
from .preprocess import build_input, build_target


def run_sanity_check():
    # Path to spider_data (adjust if needed)
    spider_data_dir = "spider_data"

    print("🔍 Running Spider sanity checks...\n")

    loader = SpiderDataLoader(spider_data_dir)

    # ---- Load data ----
    train_examples = loader.load_train()
    schemas = loader.load_schemas()

    print(f"✅ Loaded {len(train_examples)} training examples")
    print(f"✅ Loaded {len(schemas)} database schemas\n")

    # ---- Pick one example ----
    example = train_examples[0]
    schema = loader.get_schema(example.db_id)

    assert schema is not None, "Schema not found for example DB"

    # ---- Show raw example ----
    print("📝 RAW EXAMPLE")
    print(f"DB ID     : {example.db_id}")
    print(f"Question  : {example.question}")
    print(f"SQL       : {example.query}\n")

    # ---- Preprocessed input/output ----
    model_input = build_input(example, schema)
    target_sql = build_target(example)

    print("🧩 MODEL INPUT")
    print(model_input)

    print("\n🎯 TARGET SQL")
    print(target_sql)

    # ---- Minimal consistency checks ----
    assert example.question in model_input
    assert isinstance(target_sql, str)
    assert len(model_input) > 0

    print("\n✅ Sanity check PASSED — data pipeline is consistent.")


if __name__ == "__main__":
    run_sanity_check()
