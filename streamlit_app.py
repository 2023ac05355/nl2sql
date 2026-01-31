"""
Streamlit UI for Text-to-SQL Research Demonstration

This is an academic research demo, not a production application.
It demonstrates:
- Zero-shot Text-to-SQL generation using Gemini
- SQL execution on Spider databases
- Follow-up question handling via explicit SQL rewriting

All core logic is imported from existing modules - no duplication.
"""

import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

# Import existing project modules
from src.model.gemini_model import GeminiText2SQL
from src.data.load_spider import SpiderDataLoader
from src.data.preprocess import linearize_schema
from src.evaluation.sql_repair import clean_sql_output
from src.verification.syntax_check import syntax_check


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Text-to-SQL Research Demo",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Text-to-SQL Research Demonstration")
st.caption("Academic demonstration of LLM-based SQL generation with follow-up support")


# ============================================================================
# Session State Initialization
# ============================================================================

if "previous_question" not in st.session_state:
    st.session_state.previous_question = None

if "previous_sql" not in st.session_state:
    st.session_state.previous_sql = None

if "selected_db" not in st.session_state:
    st.session_state.selected_db = None

if "current_question" not in st.session_state:
    st.session_state.current_question = ""

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False


# ============================================================================
# Load Spider Data (cached)
# ============================================================================

@st.cache_resource
def load_spider_data():
    """Load Spider schemas and database list (cached for performance)."""
    loader = SpiderDataLoader("spider_data")
    schemas = loader.load_schemas()
    db_list = sorted(schemas.keys())
    return schemas, db_list


@st.cache_resource
def load_model():
    """Initialize Gemini model (cached to avoid re-initialization)."""
    return GeminiText2SQL(model_name="models/gemini-2.5-flash")


# Load data and model
schemas, db_list = load_spider_data()
model = load_model()


# ============================================================================
# Helper Functions
# ============================================================================

def build_zero_shot_prompt(question: str, schema_text: str) -> str:
    """
    Build a strict zero-shot Text-to-SQL prompt.
    
    Args:
        question: Natural language question
        schema_text: Linearized database schema
        
    Returns:
        Formatted prompt for Gemini
    """
    prompt = f"""
You are a Text-to-SQL system.

Task:
Given the database schema and the question, write a VALID SQLite SQL query.

Rules:
- Output ONLY the SQL query
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include comments
- Do NOT include multiple queries
- The query MUST be executable in SQLite

Schema:
{schema_text}

Question:
{question}

SQL:
""".strip()
    return prompt


def build_followup_prompt(
    previous_question: str,
    previous_sql: str,
    followup_question: str,
    schema_text: str
) -> str:
    """
    Build a prompt for modifying SQL based on a follow-up question.
    
    Args:
        previous_question: The original question
        previous_sql: The SQL that answered the original question
        followup_question: The follow-up modification request
        schema_text: Linearized database schema
        
    Returns:
        Formatted prompt for Gemini
    """
    prompt = f"""
You are a Text-to-SQL system that can handle follow-up questions.

Database Schema:
{schema_text}

Previous Conversation:
User: {previous_question}
SQL: {previous_sql}

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


def execute_sql_on_db(sql: str, db_path: str) -> tuple[bool, pd.DataFrame | str]:
    """
    Execute SQL on a SQLite database.
    
    Args:
        sql: SQL query to execute
        db_path: Path to SQLite database file
        
    Returns:
        Tuple of (success: bool, result: DataFrame or error message)
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return True, df
    except Exception as e:
        return False, str(e)


# ============================================================================
# UI Layout
# ============================================================================

# Sidebar: Database Selection
with st.sidebar:
    st.header("⚙️ Configuration")
    
    selected_db = st.selectbox(
        "Select Database",
        options=db_list,
        index=db_list.index("concert_singer") if "concert_singer" in db_list else 0,
        help="Choose a Spider database to query"
    )
    
    # Update session state if database changed
    if selected_db != st.session_state.selected_db:
        st.session_state.selected_db = selected_db
        st.session_state.previous_question = None
        st.session_state.previous_sql = None
        st.session_state.current_question = ""
        st.info("Database changed. Conversation cleared.")
    
    st.divider()
    
    # Display schema information
    if selected_db:
        schema = schemas[selected_db]
        st.subheader("📋 Schema Overview")
        st.caption(f"**Database:** {selected_db}")
        st.caption(f"**Tables:** {len(schema.table_names_original)} | **Columns:** {len([c for c in schema.column_names_original if c[0] != -1])}")
        
        with st.expander("View Schema Details", expanded=False):
            # Build table-column mapping
            table_data = []
            for table_idx, table_name in enumerate(schema.table_names_original):
                # Get columns for this table
                columns = [
                    col_name for t_idx, col_name in schema.column_names_original 
                    if t_idx == table_idx
                ]
                
                # Get column types for this table
                col_types = [
                    schema.column_types[i] 
                    for i, (t_idx, _) in enumerate(schema.column_names_original) 
                    if t_idx == table_idx
                ]
                
                # Find primary keys
                pk_cols = []
                for pk_idx in schema.primary_keys:
                    if pk_idx < len(schema.column_names_original):
                        t_idx, col_name = schema.column_names_original[pk_idx]
                        if t_idx == table_idx:
                            pk_cols.append(col_name)
                
                # Add row for each column
                for col_name, col_type in zip(columns, col_types):
                    is_pk = "🔑 " if col_name in pk_cols else ""
                    table_data.append({
                        "Table": table_name,
                        "Column": f"{is_pk}{col_name}",
                        "Type": col_type.upper()
                    })
            
            # Display as DataFrame
            if table_data:
                import pandas as pd
                df_schema = pd.DataFrame(table_data)
                st.dataframe(
                    df_schema,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Add foreign key information if available
                if schema.foreign_keys:
                    st.caption("**Foreign Keys:**")
                    fk_data = []
                    for from_col_idx, to_col_idx in schema.foreign_keys:
                        if from_col_idx < len(schema.column_names_original) and to_col_idx < len(schema.column_names_original):
                            from_table_idx, from_col = schema.column_names_original[from_col_idx]
                            to_table_idx, to_col = schema.column_names_original[to_col_idx]
                            
                            from_table = schema.table_names_original[from_table_idx] if from_table_idx < len(schema.table_names_original) else "?"
                            to_table = schema.table_names_original[to_table_idx] if to_table_idx < len(schema.table_names_original) else "?"
                            
                            fk_data.append({
                                "From": f"{from_table}.{from_col}",
                                "To": f"{to_table}.{to_col}"
                            })
                    
                    if fk_data:
                        df_fk = pd.DataFrame(fk_data)
                        st.dataframe(df_fk, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.previous_question = None
        st.session_state.current_question = ""
        st.session_state.previous_sql = None
        st.success("Conversation cleared!")
        st.rerun()


# Main area: Question Input and Results
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask a Question")
    
    # Show context if there's a previous question
    if st.session_state.previous_question:
        with st.container():
            st.caption("**Previous Question:**")
            st.info(st.session_state.previous_question)
            st.caption("**Previous SQL:**")
            st.code(st.session_state.previous_sql, language="sql")

with col2:
    st.subheader("📊 Query Mode")
    is_followup = st.session_state.previous_question is not None
    
    if is_followup:
        st.success("**Follow-up Mode**")
        st.caption("Your question will modify the previous SQL.")
    else:
        st.info("**New Question Mode**")
        st.caption("Starting fresh with no context.")


# Question input using form for auto-clear behavior
with st.form(key="question_form", clear_on_submit=True):
    question = st.text_input(
        "Enter your question:",
        value=st.session_state.current_question if not st.session_state.clear_input else "",
        placeholder="How many singers are there?" if not is_followup else "Show only those from France",
        help="Ask a natural language question about the database",
        key="question_input"
    )
    
    # Ask button (form submit button)
    ask_button = st.form_submit_button("🚀 Generate SQL", type="primary", use_container_width=True)


# ============================================================================
# SQL Generation and Execution
# ============================================================================

if ask_button and question:
    schema = schemas[selected_db]
    schema_text = linearize_schema(schema)
    generated_sql = ""  # Initialize to avoid unbound variable warning
    
    with st.spinner("Generating SQL..."):
        try:
            # Build appropriate prompt based on mode
            if is_followup and st.session_state.previous_question and st.session_state.previous_sql:
                prompt = build_followup_prompt(
                    previous_question=st.session_state.previous_question,
                    previous_sql=st.session_state.previous_sql,
                    followup_question=question,
                    schema_text=schema_text
                )
            else:
                prompt = build_zero_shot_prompt(question, schema_text)
            
            # Generate SQL using Gemini
            raw_sql = model.generate_sql(prompt)
            generated_sql = clean_sql_output(raw_sql)
            
        except Exception as e:
            st.error(f"❌ SQL Generation Failed: {str(e)}")
            st.stop()
    
    # Display generated SQL
    st.subheader("📝 Generated SQL")
    st.code(generated_sql, language="sql")
    
    # Check executability and execute
    db_path = Path("spider_data") / "database" / selected_db / f"{selected_db}.sqlite"
    
    if not db_path.exists():
        st.error(f"❌ Database file not found: {db_path}")
        st.stop()
    
    # Syntactic Verification Layer
    st.subheader("🔍 Syntactic Verification")
    with st.spinner("Running syntactic verification..."):
        is_valid = syntax_check(generated_sql, str(db_path))
    
    if not is_valid:
        st.error("❌ **Verification Failed**")
        st.caption("The generated SQL contains syntax errors or schema violations.")
        
        with st.expander("ℹ️ About Syntactic Verification", expanded=False):
            st.markdown("""
            **Design Philosophy:** This layer provides lightweight validation by attempting 
            to execute SQL in a read-only context. It catches:
            - Syntax errors (malformed SQL keywords/structure)
            - Schema errors (non-existent tables/columns)
            
            This is an **execution-based proxy** for syntactic checking - pragmatic for 
            academic projects where controlled risk reduction is preferred over full parsing overhead.
            """)
        
        # Still update session state for follow-up attempts
        st.session_state.previous_question = question
        st.session_state.previous_sql = generated_sql
        st.session_state.current_question = ""  # Clear for retry
        st.session_state.clear_input = True
        st.stop()
    else:
        st.success("✅ **Verification Passed** • SQL is syntactically valid")
    
    # Execute SQL
    st.subheader("📊 Execution Results")
    
    with st.spinner("Executing query..."):
        success, result = execute_sql_on_db(generated_sql, str(db_path))
    
    if success:
        # Type assertion: result is DataFrame when success is True
        assert isinstance(result, pd.DataFrame), "Expected DataFrame on success"
        
        # Display results with improved formatting
        num_rows = len(result)
        num_cols = len(result.columns) if num_rows > 0 else 0
        
        st.success(f"**Query executed successfully** • {num_rows} row{'s' if num_rows != 1 else ''} × {num_cols} column{'s' if num_cols != 1 else ''}")
        
        if num_rows == 0:
            st.info("📭 Query returned no results.")
        else:
            # Improve column names for better readability
            display_df = result.copy()
            
            # Rename generic column names to be more descriptive
            new_columns = []
            for col in display_df.columns:
                col_str = str(col)
                # Make column names more intuitive
                if col_str.lower() in ['count(*)', 'count(*)']:
                    new_columns.append('Count')
                elif col_str.lower().startswith('count('):
                    # Extract what's being counted
                    new_columns.append(f'Count of {col_str[6:-1]}')
                elif col_str.lower().startswith('avg('):
                    new_columns.append(f'Average {col_str[4:-1]}')
                elif col_str.lower().startswith('sum('):
                    new_columns.append(f'Sum of {col_str[4:-1]}')
                elif col_str.lower().startswith('max('):
                    new_columns.append(f'Maximum {col_str[4:-1]}')
                elif col_str.lower().startswith('min('):
                    new_columns.append(f'Minimum {col_str[4:-1]}')
                else:
                    # Capitalize and clean up column names
                    new_columns.append(col_str.replace('_', ' ').title())
            
            display_df.columns = new_columns
            
            # Display with enhanced styling
            st.dataframe(
                display_df, 
                use_container_width=True,
                hide_index=True
            )
            
            # Add summary for single-value results (common for aggregations)
            if num_rows == 1 and num_cols == 1:
                value = display_df.iloc[0, 0]
                col_name = display_df.columns[0]
                st.metric(
                    label=col_name,
                    value=str(value),
                    help="Single aggregate result"
                )
        
        # Update session state for follow-ups
        st.session_state.previous_question = question
        st.session_state.previous_sql = generated_sql
        st.session_state.current_question = ""  # Clear for next question
        st.session_state.clear_input = True
        
    else:
        st.error("❌ **Execution Failed**")
        st.code(result, language="text")  # Show error message
        
        # Still update session state
        st.session_state.previous_question = question
        st.session_state.previous_sql = generated_sql
        st.session_state.current_question = ""  # Clear for retry
        st.session_state.clear_input = True


# ============================================================================
# Footer / Info
# ============================================================================

st.divider()

with st.expander("ℹ️ About This Demo"):
    st.markdown("""
    ### Research Demonstration: Text-to-SQL with Follow-Up Support
    
    **Purpose:**  
    This UI demonstrates an LLM-based Text-to-SQL system with explicit follow-up question handling.
    
    **Features:**
    - **Zero-shot SQL generation** using Gemini
    - **Schema-aware prompting** from Spider dataset
    - **Follow-up question support** via explicit SQL rewriting
    - **Execution validation** on real SQLite databases
    
    **How it works:**
    1. Select a database from the sidebar
    2. Ask a natural language question
    3. The system generates SQL and executes it
    4. Ask follow-up questions to refine the query
    
    **Follow-up Mode:**  
    When you ask a follow-up question, the system receives:
    - Your previous question
    - The previous SQL query
    - Your new follow-up question
    
    It then **modifies the SQL** rather than generating from scratch.
    
    **Technical Details:**
    - Model: Gemini 2.5 Flash
    - Dataset: Spider benchmark databases
    - Temperature: 0.0 (deterministic)
    - No prompt engineering or chain-of-thought
    
    **Limitations:**
    - Follow-ups work best for simple modifications (filters, sorting)
    - Complex multi-step changes may fail
    - No query optimization or explanation
    
    ---
    
    *This is an academic research project, not a production system.*
    """)

st.caption("🔬 Text-to-SQL Research Project | Built with Streamlit")
