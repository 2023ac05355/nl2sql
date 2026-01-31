"""
Run Follow-Up Question Evaluation Experiment

This is the main script to execute the follow-up question experiment,
which tests LLM capability to modify SQL queries based on follow-up questions.

Usage:
    python run_followup_experiment.py
"""

from src.evaluation.followup_eval import run_followup_evaluation


if __name__ == "__main__":
    print("🚀 Starting Follow-Up Question Experiment\n")
    
    # Run evaluation on 20 curated examples
    results, analysis = run_followup_evaluation(
        n_examples=20,
        model_name="models/gemini-2.5-flash",
        verbose=True,
        save_output=True
    )
    
    print("\n✅ Experiment complete!")
    print("\nNext steps:")
    print("  1. Review results in experiments/ folder")
    print("  2. Analyze failure patterns")
    print("  3. Document findings in dissertation")
