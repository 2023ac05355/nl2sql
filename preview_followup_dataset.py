"""
Preview the follow-up question dataset without running evaluation.

This script displays the 20 curated follow-up examples to help understand
the experiment structure and verify the dataset quality.
"""

from src.evaluation.followup_eval import create_followup_dataset
from collections import Counter


def preview_dataset():
    """Display all follow-up examples in a readable format."""
    examples = create_followup_dataset()
    
    print("=" * 80)
    print("FOLLOW-UP QUESTION DATASET PREVIEW")
    print("=" * 80)
    print(f"\nTotal Examples: {len(examples)}")
    print(f"Database: {examples[0].db_id}")
    
    # Count modification types
    mod_types = Counter(e.modification_type for e in examples)
    print(f"\nModification Types Distribution:")
    for mod_type, count in mod_types.most_common():
        print(f"  {mod_type:30s}: {count}")
    
    print("\n" + "=" * 80)
    print("EXAMPLES")
    print("=" * 80)
    
    for example in examples:
        print(f"\n[Example {example.example_id}] - {example.modification_type}")
        print(f"  Original Q:  {example.original_question}")
        print(f"  Original SQL: {example.original_sql}")
        print(f"  Follow-up Q:  {example.followup_question}")
        print(f"  Follow-up SQL: {example.followup_sql}")
        if example.description:
            print(f"  Description: {example.description}")
        print()
    
    print("=" * 80)
    print(f"✅ Dataset contains {len(examples)} examples ready for evaluation")
    print("=" * 80)


if __name__ == "__main__":
    preview_dataset()
