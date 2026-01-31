"""
Follow-Up Question Evaluation for Text-to-SQL

This module tests whether LLMs can correctly modify a previous SQL query
in response to a follow-up question - a key capability for interactive
database querying that is not fully covered in standard benchmarks.

Research Question:
    Can an LLM correctly modify a previous SQL query in response to
    a follow-up question without re-generating from scratch?

Experiment Design:
    1. Generate initial SQL from original question
    2. Present follow-up question with context (previous Q&A)
    3. Ask model to modify the SQL
    4. Evaluate correctness of modified SQL

Metrics:
    - Executability: Can the modified SQL run without errors?
    - Execution Accuracy: Does it produce correct results vs gold follow-up?
    - Qualitative Analysis: Types of modifications (filters, aggregations, etc.)
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime

from ..model.gemini_model import GeminiText2SQL
from ..data.load_spider import SpiderDataLoader, SpiderExample, TableSchema
from ..data.preprocess import linearize_schema
from .metrics import execution_match, normalize_sql
from .sql_utils import is_executable
from .sql_repair import clean_sql_output
from ..verification.syntax_check import syntax_check


@dataclass
class FollowUpExample:
    """
    Represents a follow-up question experiment example.
    
    Attributes:
        example_id: Unique identifier for this example
        db_id: Database identifier
        original_question: Initial question
        original_sql: Gold SQL for original question
        followup_question: Follow-up question that modifies the request
        followup_sql: Gold SQL for follow-up question
        modification_type: Category of modification (filter, aggregate, join, etc.)
        description: Optional description of what changes
    """
    example_id: int
    db_id: str
    original_question: str
    original_sql: str
    followup_question: str
    followup_sql: str
    modification_type: str
    description: Optional[str] = None


def create_followup_dataset() -> List[FollowUpExample]:
    """
    Create a manually curated dataset of follow-up question pairs.
    
    These are synthetic but realistic follow-ups designed to test
    various SQL modification capabilities:
    - Adding filters (WHERE clauses)
    - Changing aggregations (COUNT → AVG, etc.)
    - Adding sorting (ORDER BY)
    - Limiting results (LIMIT)
    - Joining additional tables
    - Modifying projections (SELECT columns)
    
    Returns:
        List of FollowUpExample instances
    """
    examples = [
        # Example 1: concert_singer - Adding filter
        FollowUpExample(
            example_id=1,
            db_id="concert_singer",
            original_question="How many singers do we have?",
            original_sql="SELECT count(*) FROM singer",
            followup_question="How many of them are from France?",
            followup_sql="SELECT count(*) FROM singer WHERE country = 'France'",
            modification_type="add_filter",
            description="Adding country filter to existing count query"
        ),
        
        # Example 2: concert_singer - Changing aggregation
        FollowUpExample(
            example_id=2,
            db_id="concert_singer",
            original_question="How many concerts are there?",
            original_sql="SELECT count(*) FROM concert",
            followup_question="What's the average number of concerts per year instead?",
            followup_sql="SELECT count(*) / count(DISTINCT year) FROM concert",
            modification_type="change_aggregation",
            description="Changing from total count to average per year"
        ),
        
        # Example 3: concert_singer - Adding ORDER BY
        FollowUpExample(
            example_id=3,
            db_id="concert_singer",
            original_question="List all singer names.",
            original_sql="SELECT name FROM singer",
            followup_question="Sort them by age, oldest first.",
            followup_sql="SELECT name FROM singer ORDER BY age DESC",
            modification_type="add_sorting",
            description="Adding sort order to existing query"
        ),
        
        # Example 4: concert_singer - Adding LIMIT
        FollowUpExample(
            example_id=4,
            db_id="concert_singer",
            original_question="Show all stadium names.",
            original_sql="SELECT name FROM stadium",
            followup_question="Just show the first 5.",
            followup_sql="SELECT name FROM stadium LIMIT 5",
            modification_type="add_limit",
            description="Limiting results to top N"
        ),
        
        # Example 5: concert_singer - Changing column projection
        FollowUpExample(
            example_id=5,
            db_id="concert_singer",
            original_question="What are the names of singers?",
            original_sql="SELECT name FROM singer",
            followup_question="Show their birth year too.",
            followup_sql="SELECT name, birth_year FROM singer",
            modification_type="add_column",
            description="Adding column to SELECT clause"
        ),
        
        # Example 6: Adding numeric filter
        FollowUpExample(
            example_id=6,
            db_id="concert_singer",
            original_question="List all singers.",
            original_sql="SELECT * FROM singer",
            followup_question="Only those older than 30.",
            followup_sql="SELECT * FROM singer WHERE age > 30",
            modification_type="add_filter",
            description="Adding numeric comparison filter"
        ),
        
        # Example 7: Combining filter and sorting
        FollowUpExample(
            example_id=7,
            db_id="concert_singer",
            original_question="What concerts are there?",
            original_sql="SELECT concert_name FROM concert",
            followup_question="Show only those in 2014, sorted by name.",
            followup_sql="SELECT concert_name FROM concert WHERE year = 2014 ORDER BY concert_name",
            modification_type="add_filter_and_sort",
            description="Adding filter and sort simultaneously"
        ),
        
        # Example 8: Switching from COUNT to actual records
        FollowUpExample(
            example_id=8,
            db_id="concert_singer",
            original_question="How many stadiums are there?",
            original_sql="SELECT count(*) FROM stadium",
            followup_question="Actually, show me their names instead of counting.",
            followup_sql="SELECT name FROM stadium",
            modification_type="change_output_type",
            description="Switching from aggregation to detail view"
        ),
        
        # Example 9: Adding GROUP BY
        FollowUpExample(
            example_id=9,
            db_id="concert_singer",
            original_question="Show all concert names.",
            original_sql="SELECT concert_name FROM concert",
            followup_question="Group them by year and count each year.",
            followup_sql="SELECT year, count(*) FROM concert GROUP BY year",
            modification_type="add_grouping",
            description="Converting to grouped aggregation"
        ),
        
        # Example 10: Adding DISTINCT
        FollowUpExample(
            example_id=10,
            db_id="concert_singer",
            original_question="What countries are singers from?",
            original_sql="SELECT country FROM singer",
            followup_question="Remove duplicates.",
            followup_sql="SELECT DISTINCT country FROM singer",
            modification_type="add_distinct",
            description="Adding DISTINCT to remove duplicates"
        ),
        
        # Example 11: Multiple filters (AND condition)
        FollowUpExample(
            example_id=11,
            db_id="concert_singer",
            original_question="List singer names.",
            original_sql="SELECT name FROM singer",
            followup_question="Only females from USA.",
            followup_sql="SELECT name FROM singer WHERE gender = 'F' AND country = 'USA'",
            modification_type="add_multiple_filters",
            description="Adding compound WHERE clause"
        ),
        
        # Example 12: Changing comparison operator
        FollowUpExample(
            example_id=12,
            db_id="concert_singer",
            original_question="How many singers are older than 25?",
            original_sql="SELECT count(*) FROM singer WHERE age > 25",
            followup_question="Change it to 25 or older.",
            followup_sql="SELECT count(*) FROM singer WHERE age >= 25",
            modification_type="modify_filter",
            description="Adjusting filter condition"
        ),
        
        # Example 13: Adding JOIN
        FollowUpExample(
            example_id=13,
            db_id="concert_singer",
            original_question="List all concert names.",
            original_sql="SELECT concert_name FROM concert",
            followup_question="Include the stadium name for each concert.",
            followup_sql="SELECT concert.concert_name, stadium.name FROM concert JOIN stadium ON concert.stadium_id = stadium.stadium_id",
            modification_type="add_join",
            description="Joining additional table"
        ),
        
        # Example 14: Adding HAVING clause
        FollowUpExample(
            example_id=14,
            db_id="concert_singer",
            original_question="Count concerts by stadium.",
            original_sql="SELECT stadium_id, count(*) FROM concert GROUP BY stadium_id",
            followup_question="Only show stadiums with more than 2 concerts.",
            followup_sql="SELECT stadium_id, count(*) FROM concert GROUP BY stadium_id HAVING count(*) > 2",
            modification_type="add_having",
            description="Adding HAVING clause to grouped query"
        ),
        
        # Example 15: Inverting filter (NOT)
        FollowUpExample(
            example_id=15,
            db_id="concert_singer",
            original_question="Show singers from USA.",
            original_sql="SELECT name FROM singer WHERE country = 'USA'",
            followup_question="Actually, show everyone except USA singers.",
            followup_sql="SELECT name FROM singer WHERE country != 'USA'",
            modification_type="invert_filter",
            description="Negating filter condition"
        ),
        
        # Example 16: Adding range filter
        FollowUpExample(
            example_id=16,
            db_id="concert_singer",
            original_question="List all singers.",
            original_sql="SELECT name FROM singer",
            followup_question="Only those between ages 25 and 35.",
            followup_sql="SELECT name FROM singer WHERE age BETWEEN 25 AND 35",
            modification_type="add_range_filter",
            description="Adding BETWEEN range filter"
        ),
        
        # Example 17: Switching aggregation function
        FollowUpExample(
            example_id=17,
            db_id="concert_singer",
            original_question="How many singers are there?",
            original_sql="SELECT count(*) FROM singer",
            followup_question="What's the average age instead?",
            followup_sql="SELECT avg(age) FROM singer",
            modification_type="switch_aggregation",
            description="Replacing COUNT with AVG on different column"
        ),
        
        # Example 18: Adding both LIMIT and ORDER BY
        FollowUpExample(
            example_id=18,
            db_id="concert_singer",
            original_question="Show all singers.",
            original_sql="SELECT name FROM singer",
            followup_question="Show only the 3 youngest.",
            followup_sql="SELECT name FROM singer ORDER BY age ASC LIMIT 3",
            modification_type="add_top_n",
            description="Adding ORDER BY and LIMIT for top N query"
        ),
        
        # Example 19: Removing filter
        FollowUpExample(
            example_id=19,
            db_id="concert_singer",
            original_question="How many singers from France?",
            original_sql="SELECT count(*) FROM singer WHERE country = 'France'",
            followup_question="Actually show all countries.",
            followup_sql="SELECT count(*) FROM singer",
            modification_type="remove_filter",
            description="Removing WHERE clause"
        ),
        
        # Example 20: Complex multi-step modification
        FollowUpExample(
            example_id=20,
            db_id="concert_singer",
            original_question="List all singers.",
            original_sql="SELECT name FROM singer",
            followup_question="Show name and age for singers over 40, sorted by age descending, limit 10.",
            followup_sql="SELECT name, age FROM singer WHERE age > 40 ORDER BY age DESC LIMIT 10",
            modification_type="complex_modification",
            description="Adding column, filter, sort, and limit simultaneously"
        ),
    ]
    
    return examples


def build_followup_prompt(
    original_question: str,
    original_sql: str,
    followup_question: str,
    schema: TableSchema
) -> str:
    """
    Build a prompt for the LLM to modify SQL based on a follow-up question.
    
    This prompt provides:
    - Database schema
    - Original question and its SQL
    - Follow-up question
    - Instruction to modify the SQL
    
    Args:
        original_question: The initial natural language question
        original_sql: The SQL query that answered the original question
        followup_question: The follow-up modification request
        schema: Database schema
        
    Returns:
        Formatted prompt string for the LLM
    """
    schema_text = linearize_schema(schema)
    
    prompt = f"""
You are a Text-to-SQL system that can handle follow-up questions.

Database Schema:
{schema_text}

Previous Conversation:
User: {original_question}
SQL: {original_sql}

Follow-Up Question:
{followup_question}

Task:
Modify the previous SQL query to answer the follow-up question.

Rules:
- Output ONLY the modified SQL query
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include comments
- The query MUST be executable in SQLite
- Use the previous SQL as a starting point

Modified SQL:
""".strip()
    
    return prompt


@dataclass
class FollowUpResult:
    """
    Results from a single follow-up evaluation.
    """
    example_id: int
    db_id: str
    modification_type: str
    original_question: str
    original_sql: str
    followup_question: str
    gold_followup_sql: str
    predicted_followup_sql: str
    is_executable: bool
    execution_match: bool
    error_message: Optional[str] = None


def evaluate_followup_examples(
    examples: List[FollowUpExample],
    model: GeminiText2SQL,
    spider_loader: SpiderDataLoader,
    verbose: bool = True
) -> List[FollowUpResult]:
    """
    Evaluate LLM performance on follow-up question modifications.
    
    Args:
        examples: List of follow-up examples to evaluate
        model: Text-to-SQL model to test
        spider_loader: Spider data loader for schemas and databases
        verbose: Whether to print progress
        
    Returns:
        List of evaluation results
    """
    schemas = spider_loader.load_schemas()
    results = []
    
    for i, example in enumerate(examples):
        if verbose:
            print(f"\n[{i+1}/{len(examples)}] Evaluating example {example.example_id}...")
            print(f"  DB: {example.db_id}")
            print(f"  Type: {example.modification_type}")
        
        # Get schema
        schema = schemas[example.db_id]
        
        # Build prompt with context
        prompt = build_followup_prompt(
            original_question=example.original_question,
            original_sql=example.original_sql,
            followup_question=example.followup_question,
            schema=schema
        )
        
        # Generate modified SQL
        try:
            raw_output = model.generate_sql(prompt)
            predicted_sql = clean_sql_output(raw_output)
        except Exception as e:
            results.append(FollowUpResult(
                example_id=example.example_id,
                db_id=example.db_id,
                modification_type=example.modification_type,
                original_question=example.original_question,
                original_sql=example.original_sql,
                followup_question=example.followup_question,
                gold_followup_sql=example.followup_sql,
                predicted_followup_sql="",
                is_executable=False,
                execution_match=False,
                error_message=f"Model generation error: {str(e)}"
            ))
            continue
        
        # Get database path
        db_path = Path("spider_data") / "database" / example.db_id / f"{example.db_id}.sqlite"
        
        # Syntactic Verification Layer: Check syntax validity
        executable = syntax_check(predicted_sql, str(db_path))
        
        # Check execution accuracy (only if executable)
        exec_match = False
        error_msg = None
        
        if executable:
            exec_match = execution_match(
                predicted_sql,
                example.followup_sql,
                str(db_path)
            )
            if verbose:
                print(f"  ✓ Executable: Yes")
                print(f"  ✓ Execution Match: {exec_match}")
        else:
            error_msg = "SQL not executable"
            if verbose:
                print(f"  ✗ Executable: No")
        
        if verbose and i < 3:
            print(f"  Predicted SQL: {predicted_sql}")
            print(f"  Gold SQL:      {example.followup_sql}")
        
        # Store result
        results.append(FollowUpResult(
            example_id=example.example_id,
            db_id=example.db_id,
            modification_type=example.modification_type,
            original_question=example.original_question,
            original_sql=example.original_sql,
            followup_question=example.followup_question,
            gold_followup_sql=example.followup_sql,
            predicted_followup_sql=predicted_sql,
            is_executable=executable,
            execution_match=exec_match,
            error_message=error_msg
        ))
    
    return results


def analyze_results(results: List[FollowUpResult]) -> Dict:
    """
    Analyze and aggregate results from follow-up evaluation.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary containing aggregated metrics and analysis
    """
    total = len(results)
    executable_count = sum(1 for r in results if r.is_executable)
    exec_match_count = sum(1 for r in results if r.execution_match)
    
    # Per-modification-type breakdown
    type_stats = {}
    for result in results:
        mod_type = result.modification_type
        if mod_type not in type_stats:
            type_stats[mod_type] = {
                "total": 0,
                "executable": 0,
                "correct": 0
            }
        type_stats[mod_type]["total"] += 1
        if result.is_executable:
            type_stats[mod_type]["executable"] += 1
        if result.execution_match:
            type_stats[mod_type]["correct"] += 1
    
    analysis = {
        "summary": {
            "total_examples": total,
            "executable": executable_count,
            "execution_accuracy": exec_match_count,
            "executability_rate": executable_count / total if total > 0 else 0,
            "execution_accuracy_rate": exec_match_count / total if total > 0 else 0,
            "execution_accuracy_among_executable": exec_match_count / executable_count if executable_count > 0 else 0
        },
        "by_modification_type": type_stats,
        "failed_examples": [
            {
                "example_id": r.example_id,
                "modification_type": r.modification_type,
                "followup_question": r.followup_question,
                "error": r.error_message
            }
            for r in results if not r.execution_match
        ]
    }
    
    return analysis


def print_results(analysis: Dict) -> None:
    """
    Print formatted evaluation results.
    """
    summary = analysis["summary"]
    
    print("\n" + "="*70)
    print("📊 FOLLOW-UP QUESTION EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nOverall Performance:")
    print(f"  Total Examples:           {summary['total_examples']}")
    print(f"  Executable:               {summary['executable']}/{summary['total_examples']} ({summary['executability_rate']:.1%})")
    print(f"  Execution Accuracy:       {summary['execution_accuracy']}/{summary['total_examples']} ({summary['execution_accuracy_rate']:.1%})")
    print(f"  Accuracy (among valid):   {summary['execution_accuracy']}/{summary['executable']} ({summary['execution_accuracy_among_executable']:.1%})")
    
    print(f"\n\nBy Modification Type:")
    print("-" * 70)
    type_stats = analysis["by_modification_type"]
    for mod_type, stats in sorted(type_stats.items()):
        total = stats["total"]
        correct = stats["correct"]
        executable = stats["executable"]
        print(f"  {mod_type:25s}  {correct}/{total} correct  ({executable}/{total} executable)")
    
    if analysis["failed_examples"]:
        print(f"\n\nFailed Examples ({len(analysis['failed_examples'])}):")
        print("-" * 70)
        for fail in analysis["failed_examples"][:5]:  # Show first 5
            print(f"  [{fail['example_id']}] {fail['modification_type']}")
            print(f"      Q: {fail['followup_question']}")
            if fail['error']:
                print(f"      Error: {fail['error']}")
    
    print("\n" + "="*70)


def save_results(results: List[FollowUpResult], analysis: Dict, output_dir: str = "experiments") -> None:
    """
    Save detailed results and analysis to JSON files.
    
    Args:
        results: List of individual evaluation results
        analysis: Aggregated analysis dictionary
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = output_path / f"followup_results_{timestamp}.json"
    results_data = [
        {
            "example_id": r.example_id,
            "db_id": r.db_id,
            "modification_type": r.modification_type,
            "original_question": r.original_question,
            "original_sql": r.original_sql,
            "followup_question": r.followup_question,
            "gold_followup_sql": r.gold_followup_sql,
            "predicted_followup_sql": r.predicted_followup_sql,
            "is_executable": r.is_executable,
            "execution_match": r.execution_match,
            "error_message": r.error_message
        }
        for r in results
    ]
    
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    
    # Save analysis summary
    summary_file = output_path / f"followup_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n💾 Results saved to:")
    print(f"  - {results_file}")
    print(f"  - {summary_file}")


def run_followup_evaluation(
    n_examples: int = 20,
    model_name: str = "models/gemini-2.5-flash",
    verbose: bool = True,
    save_output: bool = True
) -> Tuple[List[FollowUpResult], Dict]:
    """
    Run complete follow-up question evaluation experiment.
    
    Args:
        n_examples: Number of examples to evaluate (default: 20)
        model_name: Gemini model to use
        verbose: Whether to print progress
        save_output: Whether to save results to disk
        
    Returns:
        Tuple of (results, analysis)
    """
    if verbose:
        print("🧪 Follow-Up Question Evaluation Experiment")
        print("=" * 70)
        print(f"Model: {model_name}")
        print(f"Examples: {n_examples}")
        print()
    
    # Initialize components
    model = GeminiText2SQL(model_name=model_name)
    loader = SpiderDataLoader("spider_data")
    
    # Create dataset
    all_examples = create_followup_dataset()
    examples = all_examples[:n_examples]
    
    if verbose:
        print(f"📝 Created {len(examples)} follow-up examples")
        print(f"Modification types: {len(set(e.modification_type for e in examples))}")
    
    # Run evaluation
    results = evaluate_followup_examples(examples, model, loader, verbose=verbose)
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print results
    if verbose:
        print_results(analysis)
    
    # Save to disk
    if save_output:
        save_results(results, analysis)
    
    return results, analysis


if __name__ == "__main__":
    # Run evaluation on 20 examples
    run_followup_evaluation(n_examples=20, verbose=True, save_output=True)
