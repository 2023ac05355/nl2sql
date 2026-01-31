# System Configuration Freeze

**Date:** January 27, 2026

## Configuration

- **Model:** Gemini-1.5-Flash
- **Dataset:** Spider (dev, N=50)
- **Metrics:** EM, EX, Syntactic Failure Rate

## System Layers

- Prompting
- Syntactic Verification
- Execution Validation
- Follow-up SQL Rewriting

## UI

- **Streamlit** (visualization only)
2. **EX (Execution Accuracy):** Result-level equality on gold database
3. **Invalid SQL Rate:** Percentage of non-executable queries

**Normalization:**
- Lowercase conversion
- Whitespace compression
- Semicolon removal

---

## Interaction Model

**Type:** SQL-grounded follow-up rewriting

**Mechanism:**
- Previous question + previous SQL + follow-up question → Modified SQL
- Explicit context passing (not implicit chat history)
- Model instructed to **modify** previous SQL, not regenerate

**Follow-Up Prompt Structure:**
```
Database Schema: [schema]

Previous Conversation:
User: [previous question]
SQL: [previous SQL]

Follow-Up Question: [follow-up]

Task: Modify the previous SQL query to answer the follow-up question.

Rules: [same as zero-shot]
```

**Key Difference from Baselines:**
- Standard benchmarks: Single-turn generation
- This system: Contextual SQL modification

---

## User Interface

**Framework:** Streamlit  
**Purpose:** Visualization and demonstration only (not production)

**Features:**
- Database selector (166 Spider databases)
- Schema viewer (tabular format with primary keys)
- Question input with auto-clear
- Follow-up mode with context banner
- SQL execution and results display
- Metric cards for aggregate results
- Error handling with validation

**Note:** UI is for qualitative analysis only. Automated evaluation uses separate scripts.

---

## Dataset

**Source:** Spider v1.0  
**Databases:** 166 training + 20 development  
**Evaluation Set:** Spider dev.json (1,034 examples)  
**Follow-Up Evaluation:** 20 manually curated pairs (concert_singer database)

---

## Code Structure

```
src/
  model/
    gemini_model.py          # Model wrapper
  data/
    load_spider.py           # Dataset loader
    preprocess.py            # Schema linearization, prompt building
  evaluation/
    metrics.py               # EM, EX calculation
    sql_utils.py             # Executability checking
    followup_eval.py         # Follow-up experiment
streamlit_app.py             # UI (demo only)
```

---

## Reproducibility

**Environment:**
- Python 3.12.10
- Virtual environment: `.venv`
- Dependencies: `requriements.txt`

**Key Dependencies:**
- `google-generativeai` (Gemini API)
- `streamlit` (UI)
- `pandas` (data handling)
- `sqlite3` (built-in, SQL execution)

**API Key:** Loaded from `.env` file (not version controlled)

---

## Limitations Acknowledged

1. **Single Model:** Only Gemini-2.5-Flash tested
2. **Small Follow-Up Dataset:** 20 examples (pilot scale)
3. **Single Domain:** Follow-ups based on concert_singer only
4. **Synthetic Follow-Ups:** Manually designed, not from real users
5. **No User Study:** Automated evaluation only

---

## Baseline Comparison

**Baseline Results (Gemini-2.5-Flash, 50 Spider dev samples):**
- EM: 0.28 (28%)
- EX: 0.80 (80%)
- Invalid SQL: 0.02 (2%)

**Follow-Up Results:**
- [To be filled after running experiment]
- Expected EX: 60-75% (harder task than baseline)

---

## Defense Points

If challenged on:

**"Why Gemini and not GPT-4?"**
→ Gemini was accessible, consistent, and sufficient for demonstrating the follow-up mechanism. The contribution is the evaluation framework, not model comparison.

**"Why only 20 follow-up examples?"**
→ Pilot study scale, sufficient for demonstrating feasibility. Expanding to 50-100 examples is listed as future work.

**"Why synthetic follow-ups?"**
→ Existing benchmarks (SParC, CoSQL) don't isolate SQL modification. Synthetic examples provide controlled evaluation of specific modification types.

**"Why Streamlit UI?"**
→ Research demonstration tool, not a production interface. Enables qualitative validation of automated results.

**"Why zero-shot only?"**
→ Most comparable to existing benchmark protocols. Few-shot would introduce prompt engineering confounds.

---

## Future Extensions (Out of Scope)

- [ ] Multi-model comparison (GPT-4, Claude, Llama)
- [ ] Larger follow-up dataset (50-100 examples)
- [ ] Real user study with natural follow-ups
- [ ] Multi-turn chains (3+ questions)
- [ ] Cross-database follow-ups
- [ ] Few-shot prompt comparison
- [ ] Retrieval-augmented generation

---

## Frozen Configuration Summary

| Component | Value |
|-----------|-------|
| Model | Gemini-2.5-Flash |
| Prompt | Zero-shot SQL-only |
| Temperature | 0.0 |
| Dataset | Spider v1.0 |
| Eval Metrics | EM, EX, Invalid Rate |
| Interaction | SQL-grounded follow-up |
| UI | Streamlit (demo) |
| Date Frozen | January 27, 2026 |

---

**This configuration is now frozen for dissertation submission.**  
**Any changes after this point must be documented as modifications or extensions.**

---

*Protects against: dependency updates, API changes, breaking code edits, scope creep*
