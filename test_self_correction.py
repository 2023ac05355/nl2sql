"""
Test script for Automated Self-Correction Loop

This script demonstrates and tests the self-correction capability
by intentionally creating scenarios where SQL might fail and showing
how the system recovers.

Usage:
    python test_self_correction.py
"""

import os
from pathlib import Path
from src.model.gemini_model import GeminiText2SQL
from src.data.load_spider import SpiderDataLoader
from src.data.preprocess import linearize_schema
from src.evaluation.metrics import execute_sql, execution_match
from src.evaluation.sql_repair import clean_sql_output


def test_self_correction_basic():
    """
    Test basic self-correction functionality with a simple example.
    """
    print("=" * 70)
    print("TEST 1: Basic Self-Correction with Intentional Error")
    print("=" * 70)
    
    # Setup
    loader = SpiderDataLoader("spider_data")
    schemas = loader.load_schemas()
    model = GeminiText2SQL()
    
    # Use concert_singer database
    db_id = "concert_singer"
    schema = schemas[db_id]
    schema_text = linearize_schema(schema)
    db_path = Path("spider_data") / "database" / db_id / f"{db_id}.sqlite"
    
    # Test case: SQL with intentional error (wrong table name)
    question = "How many singers are there?"
    failed_sql = "SELECT count(*) FROM singerz"  # Wrong table name
    
    print(f"\nQuestion: {question}")
    print(f"Failed SQL: {failed_sql}")
    
    try:
        result, recovered = execute_sql(
            db_path=str(db_path),
            sql=failed_sql,
            model=model,
            question=question,
            schema_text=schema_text
        )
        
        if recovered:
            print("\n✅ RECOVERY SUCCESSFUL!")
            print(f"Result: {result}")
        else:
            print("\n✅ Executed without recovery (SQL was correct)")
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"\n❌ RECOVERY FAILED: {e}")
    
    print()


def test_self_correction_with_benchmark():
    """
    Test self-correction on actual Spider dev examples.
    """
    print("=" * 70)
    print("TEST 2: Self-Correction on Spider Dev Examples")
    print("=" * 70)
    
    # Setup
    loader = SpiderDataLoader("spider_data")
    dev_examples = loader.load_dev()
    schemas = loader.load_schemas()
    model = GeminiText2SQL()
    
    n_samples = 10  # Test on first 10 examples
    recovered_count = 0
    total_executions = 0
    match_count = 0
    
    print(f"\nTesting on {n_samples} examples...\n")
    
    for i, example in enumerate(dev_examples[:n_samples]):
        print(f"Example {i+1}/{n_samples}: {example.db_id}")
        
        schema = schemas[example.db_id]
        schema_text = linearize_schema(schema)
        db_path = Path("spider_data") / "database" / example.db_id / f"{example.db_id}.sqlite"
        
        # Generate SQL
        from src.data.preprocess import build_gemini_zeroshot_input
        prompt = build_gemini_zeroshot_input(example, schema)
        
        try:
            raw_sql = model.generate_sql(prompt)
            pred_sql = clean_sql_output(raw_sql)
            
            # Test with self-correction
            match, recovered = execution_match(
                pred_sql=pred_sql,
                gold_sql=example.query,
                db_path=str(db_path),
                model=model,
                question=example.question,
                schema_text=schema_text
            )
            
            total_executions += 1
            
            if recovered:
                recovered_count += 1
                print(f"  🔄 RECOVERED from error")
            
            if match:
                match_count += 1
                print(f"  ✅ Execution match: YES")
            else:
                print(f"  ❌ Execution match: NO")
                
        except Exception as e:
            print(f"  ❌ Failed: {str(e)[:50]}...")
        
        print()
    
    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total examples tested: {n_samples}")
    print(f"Successful executions: {total_executions}")
    print(f"Recovered from errors: {recovered_count}")
    print(f"Execution matches: {match_count}")
    
    if total_executions > 0:
        print(f"\nRecovery rate: {recovered_count / total_executions:.1%}")
        print(f"Match rate: {match_count / total_executions:.1%}")
    
    print()


def test_manual_correction_example():
    """
    Manually test a specific error scenario to verify correction logic.
    """
    print("=" * 70)
    print("TEST 3: Manual Error Correction Example")
    print("=" * 70)
    
    # Setup
    loader = SpiderDataLoader("spider_data")
    schemas = loader.load_schemas()
    model = GeminiText2SQL()
    
    db_id = "concert_singer"
    schema = schemas[db_id]
    schema_text = linearize_schema(schema)
    db_path = Path("spider_data") / "database" / db_id / f"{db_id}.sqlite"
    
    # Test different error types
    test_cases = [
        {
            "question": "How many singers are there?",
            "failed_sql": "SELECT count(*) FROM wrong_table",
            "error_type": "Table doesn't exist"
        },
        {
            "question": "List all singer names",
            "failed_sql": "SELECT wrong_column FROM singer",
            "error_type": "Column doesn't exist"
        },
        {
            "question": "Show stadium names",
            "failed_sql": "SELECT name FORM stadium",  # Typo: FORM instead of FROM
            "error_type": "Syntax error"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['error_type']}")
        print(f"Question: {test['question']}")
        print(f"Failed SQL: {test['failed_sql']}")
        
        try:
            result, recovered = execute_sql(
                db_path=str(db_path),
                sql=test['failed_sql'],
                model=model,
                question=test['question'],
                schema_text=schema_text
            )
            
            if recovered:
                print(f"  ✅ RECOVERED! Result: {result}")
            else:
                print(f"  ✅ No recovery needed. Result: {result}")
                
        except Exception as e:
            print(f"  ❌ Recovery failed: {str(e)[:100]}")
        
    print()


if __name__ == "__main__":
    print("\n🧪 TESTING AUTOMATED SELF-CORRECTION LOOP\n")
    
    # Check if API key is set
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Warning: GEMINI_API_KEY not set in environment")
        print("   Set it with: $env:GEMINI_API_KEY = 'your-api-key'\n")
    
    # Run tests
    try:
        print("Running Test Suite...\n")
        
        # Test 1: Basic functionality
        test_self_correction_basic()
        
        # Test 2: Real examples from Spider
        test_self_correction_with_benchmark()
        
        # Test 3: Specific error scenarios
        test_manual_correction_example()
        
        print("=" * 70)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Review recovery patterns in the output above")
        print("  2. Run full benchmark with: python -m src.evaluation.run_benchmark")
        print("  3. Document recovery rate in your dissertation")
        print()
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
