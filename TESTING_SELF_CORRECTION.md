# Testing the Automated Self-Correction Loop

## Quick Start

### 1. Set up your environment
```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Set API key
$env:GEMINI_API_KEY = "your-api-key-here"
```

### 2. Run the dedicated test script
```powershell
python test_self_correction.py
```

**What it does:**
- Tests basic self-correction with intentional errors
- Runs on 10 Spider dev examples
- Shows manual error correction for different error types
- Displays recovery statistics

**Expected output:**
```
✅ RECOVERY SUCCESSFUL!
🔄 RECOVERED from error
Recovery rate: X%
```

### 3. Run the full benchmark (with self-correction enabled)
```powershell
python -m src.evaluation.run_benchmark
```

**What changed:**
- Now tracks recovered queries
- Shows recovery rate and contribution to execution accuracy
- Displays first 5 recoveries in real-time

**New metrics in output:**
```
🔄 SELF-CORRECTION STATISTICS
Recovered queries: X
Recovery rate: 0.XXX
Recovery contribution to EX: XX%
```

### 4. Compare with/without self-correction
```python
# In Python console or script
from src.evaluation.run_benchmark import run_benchmark

# With self-correction (default)
run_benchmark(n_samples=50, enable_self_correction=True)

# Without self-correction (baseline)
run_benchmark(n_samples=50, enable_self_correction=False)
```

## What to Look For

### Success Indicators
✅ **Recovered queries > 0** - Self-correction is working  
✅ **Recovery rate > 0%** - System is fixing some errors  
✅ **EX increases** - Execution accuracy improved by recovery  

### In Your Dissertation
Document these metrics:
- **Initial failure rate** (queries that fail initially)
- **Recovery success rate** (% of failures that were fixed)
- **Overall EX improvement** (with vs without self-correction)
- **Error types recovered** (table names, columns, syntax)

## Example Test Cases

The test script includes:

1. **Wrong table name**: `SELECT * FROM singerz` → `SELECT * FROM singer`
2. **Wrong column name**: `SELECT wrong_column FROM singer` → `SELECT name FROM singer`
3. **Syntax error**: `SELECT name FORM stadium` → `SELECT name FROM stadium`

## Troubleshooting

**If no recoveries occur:**
- Your initial SQL generation might be very accurate (good!)
- Try testing with intentionally broken SQL to verify the mechanism

**If recovery fails:**
- Check that API key is set correctly
- Verify the error message is being captured
- Look at the refinement prompt being sent

**If tests fail to run:**
```powershell
# Install dependencies
pip install google-genai pandas

# Check Python environment
python --version  # Should be 3.12+
```

## Advanced: Track Individual Recoveries

To see which specific queries were recovered, modify the benchmark script to log details:

```python
if recovered:
    print(f"Question: {example.question}")
    print(f"Original SQL: {pred_sql}")
    print(f"Status: RECOVERED")
```

## Next Steps for Dissertation

1. **Run baseline**: Document performance WITHOUT self-correction
2. **Run with self-correction**: Document improved performance
3. **Analyze patterns**: What types of errors get fixed?
4. **Calculate improvement**: (EX_with - EX_without) / EX_without
5. **Discuss limitations**: Single retry, API cost, latency

## Performance Notes

- **Latency**: Recovery adds 1 extra API call per failed query
- **Cost**: ~2x API calls for queries that fail initially
- **Accuracy gain**: Typically 5-15% improvement in execution accuracy
