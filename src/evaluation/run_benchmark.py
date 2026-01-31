"""
Run baseline Text-to-SQL benchmark on Spider dev set.
"""

from ..model.base_model import BaseText2SQLModel
from ..data.load_spider import SpiderDataLoader
from ..data.preprocess import build_input, build_gemini_zeroshot_input
from .metrics import exact_match, execution_match
from pathlib import Path
from .sql_utils import is_executable
from .sql_repair import extract_select_statement, clean_sql_output
from ..model.ollama_model import OllamaText2SQL
from ..model.gemini_model import GeminiText2SQL
from ..verification.syntax_check import syntax_check


demos = [
    (
        "How many heads are older than 56?",
        "head(age)",
        "SELECT count(*) FROM head WHERE age > 56"
    ),
    (
        "List all departments.",
        "department(dept_id, name)",
        "SELECT name FROM department"
    )
]


def run_benchmark(n_samples: int = 100):
    loader = SpiderDataLoader("spider_data")
    dev_examples = loader.load_dev()
    schemas = loader.load_schemas()

    # model = OllamaText2SQL("llama3.1:8b")
    model = GeminiText2SQL()


    em_count = 0
    ex_count = 0
    invalid_sql = 0

    for i, example in enumerate(dev_examples[:n_samples]):

        schema = schemas[example.db_id]
        model_input = build_gemini_zeroshot_input(example, schema)

        raw_sql = model.generate_sql(model_input)
        pred_sql = clean_sql_output(raw_sql)


        gold_sql = example.query

        db_path = Path("spider_data") / "database" / example.db_id / f"{example.db_id}.sqlite"

        # 1. Syntactic Verification Layer: Check syntax validity
        if not syntax_check(pred_sql, str(db_path)):
            invalid_sql += 1
            continue  # skip EM / EX checks - SQL failed lightweight verification

        # 2. Exact Match (string-level)
        if exact_match(pred_sql, gold_sql):
            em_count += 1

        # 3. Execution Accuracy
        if execution_match(pred_sql, gold_sql, str(db_path)):
            ex_count += 1
        if i < 3:
            print("\n--- RAW GEMINI OUTPUT ---")
            print(pred_sql)



    print("\n📊 BENCHMARK RESULTS")
    print(f"Samples evaluated: {n_samples}")
    print(f"Exact Match (EM): {em_count / n_samples:.3f}")
    print(f"Execution Accuracy (EX): {ex_count / n_samples:.3f}")
    print(f"Invalid / Failed SQL: {invalid_sql / n_samples:.3f}")


if __name__ == "__main__":
    run_benchmark()
