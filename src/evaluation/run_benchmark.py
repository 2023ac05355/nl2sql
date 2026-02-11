"""
Run baseline Text-to-SQL benchmark on Spider dev set.

Includes automated self-correction loop for failed SQL queries.
"""

from ..data.load_spider import SpiderDataLoader
from ..data.preprocess import build_input, build_gemini_zeroshot_input, linearize_schema
from .metrics import exact_match, execution_match
from pathlib import Path
from .sql_utils import is_executable
from .sql_repair import extract_select_statement, clean_sql_output
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


def run_benchmark(
    n_samples: int = 100, 
    enable_self_correction: bool = True,
    max_retries: int = 2,
    debug: bool = False
):
    """
    Run benchmark evaluation on Spider dev set.
    
    Args:
        n_samples: Number of examples to evaluate
        enable_self_correction: If True, uses automated self-correction loop
        max_retries: Maximum number of retry attempts per query
        debug: If True, print detailed debugging information for failed queries
    """
    loader = SpiderDataLoader("spider_data")
    dev_examples = loader.load_dev()
    schemas = loader.load_schemas()

    # model = OllamaText2SQL("llama3.1:8b")
    model = GeminiText2SQL()

    em_count = 0
    ex_count = 0
    syntax_failed = 0
    recovered_count = 0
    recovery_attempts = 0

    print(f"\n🚀 Starting benchmark with {n_samples} samples")
    print(f"   Self-correction: {'Enabled' if enable_self_correction else 'Disabled'}")
    if enable_self_correction:
        print(f"   Max retries: {max_retries}")
        print(f"   Debug mode: {'ON' if debug else 'OFF'}")
    print()

    for i, example in enumerate(dev_examples[:n_samples]):

        schema = schemas[example.db_id]
        model_input = build_gemini_zeroshot_input(example, schema)

        raw_sql = model.generate_sql(model_input)
        pred_sql = clean_sql_output(raw_sql)

        gold_sql = example.query

        db_path = Path("spider_data") / "database" / example.db_id / f"{example.db_id}.sqlite"

        # 1. Syntactic Verification Layer: Check syntax validity
        syntax_valid = syntax_check(pred_sql, str(db_path))
        
        if not syntax_valid and not enable_self_correction:
            # Legacy mode: just skip
            syntax_failed += 1
            continue

        # 2. Exact Match (string-level) - only if syntax is valid
        if syntax_valid and exact_match(pred_sql, gold_sql):
            em_count += 1

        # 3. Execution Accuracy (with optional self-correction)
        if enable_self_correction:
            schema_text = linearize_schema(schema)
            
            # Enable debug for ALL failures when debug is on, not just first 5
            show_debug = debug and (not syntax_valid or recovery_attempts < 10)
            
            # Try execution with self-correction enabled
            match, recovered = execution_match(
                pred_sql, 
                gold_sql, 
                str(db_path),
                model=model,
                question=example.question,
                schema_text=schema_text,
                max_retries=max_retries,
                debug=show_debug,
                enable_semantic_correction=True  # Explicitly enable
            )
            
            if match:
                ex_count += 1
                
            if recovered:
                recovered_count += 1
                recovery_attempts += 1  # Count as a recovery attempt
                print(f"\n🎉 Example {i+1}: RECOVERED!")
                print(f"   Question: {example.question[:70]}...")
                print(f"   Database: {example.db_id}")
                print()
            elif not syntax_valid or not match:
                # Failed and NOT recovered
                recovery_attempts += 1
                
            if not match and not recovered:
                syntax_failed += 1
                
        else:
            # Legacy mode without self-correction
            if syntax_valid and execution_match(pred_sql, gold_sql, str(db_path), max_retries=0)[0]:
                ex_count += 1
            elif not syntax_valid:
                syntax_failed += 1
                
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"📊 Progress: {i+1}/{n_samples} samples evaluated...")

    print("\n" + "="*70)
    print("📊 BENCHMARK RESULTS")
    print("="*70)
    print(f"Samples evaluated: {n_samples}")
    print(f"Exact Match (EM): {em_count / n_samples:.3f} ({em_count}/{n_samples})")
    print(f"Execution Accuracy (EX): {ex_count / n_samples:.3f} ({ex_count}/{n_samples})")
    print(f"Failed SQL (final): {syntax_failed / n_samples:.3f} ({syntax_failed}/{n_samples})")
    
    if enable_self_correction:
        print("\n" + "="*70)
        print("🔄 SELF-CORRECTION STATISTICS")
        print("="*70)
        print(f"Recovery attempts: {recovery_attempts}")
        print(f"Successful recoveries: {recovered_count}")
        if recovery_attempts > 0:
            print(f"Recovery success rate: {recovered_count / recovery_attempts:.1%}")
            print(f"Contribution to final EX: {recovered_count / n_samples:.1%}")
            print(f"\n💡 Insight: Self-correction improved EX by {recovered_count} queries")
            print(f"   Without self-correction, EX would be: {(ex_count - recovered_count) / n_samples:.3f}")
        else:
            print("No recovery attempts needed (all queries succeeded on first try)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Text-to-SQL benchmark on Spider dev set"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of examples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--enable-self-correction",
        action="store_true",
        default=True,
        help="Enable automated self-correction loop (default: True)"
    )
    parser.add_argument(
        "--no-self-correction",
        action="store_false",
        dest="enable_self_correction",
        help="Disable self-correction (baseline mode)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retry attempts per query (default: 2)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for first 5 recovery attempts"
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        n_samples=args.n_samples,
        enable_self_correction=args.enable_self_correction,
        max_retries=args.max_retries,
        debug=args.debug
    )
