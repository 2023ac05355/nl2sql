# Final Results

| Metric | Value |
|--------|-------|
| Exact Match (EM) | 0.28 |
| Execution Accuracy (EX) | 0.80 |
| Failed Syntax Validation | 0.02 |

## Interpretation

**Execution Accuracy (EX)** measures whether the generated SQL produces the same result set as the gold query when executed on the database, accounting for functionally equivalent queries with different syntax. **Exact Match (EM)** is lower because it requires string-level equality after normalization, missing semantically correct queries that use alternative table joins, column orderings, or aggregation patterns. The **syntax validation layer** successfully filters out 2% of malformed queries before execution, providing controlled risk reduction with minimal overhead while maintaining high execution accuracy.
