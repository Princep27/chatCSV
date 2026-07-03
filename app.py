
import os
import time

import duckdb
import pandas as pd
import plotly.express as px
import sqlglot
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
load_dotenv()

# LLM_PROVIDER selects which LangChain chat-model integration to use.
# LLM_MODEL is just the model name passed to that provider.
# Both are read purely from env vars, so changing provider/model never
# requires touching code - only your .env file.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_SAMPLE_ROWS = 5          # rows per table sent to the LLM (token control)
ALLOWED_EXTENSIONS = (".csv", ".xlsx")

st.set_page_config(page_title="DataCopilot AI", page_icon="🦆", layout="wide")
st.title("🦆 chatCSV")
st.caption("Upload multiple files, ask questions in plain English, get SQL + answers + charts.")


def get_llm():
    """
    Provider-agnostic LLM factory. Every branch uses a LangChain chat-model
    integration (never a raw provider SDK directly). To add a new provider,
    add one branch here - the rest of the app never changes.

    Switch providers/models purely via .env:
        LLM_PROVIDER=openai      LLM_MODEL=gpt-4o-mini
        LLM_PROVIDER=anthropic   LLM_MODEL=claude-3-5-sonnet-20241022
        LLM_PROVIDER=groq        LLM_MODEL=llama-3.1-70b-versatile
        LLM_PROVIDER=ollama      LLM_MODEL=llama3.1   (local, no API key)
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY is not set. Add it to your .env file.")
            st.stop()
        return ChatOpenAI(model=LLM_MODEL, api_key=api_key, temperature=0)

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
            st.stop()
        return ChatAnthropic(model=LLM_MODEL, api_key=api_key, temperature=0)

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY is not set. Add it to your .env file.")
            st.stop()
        return ChatGroq(model=LLM_MODEL, api_key=api_key, temperature=0)

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=LLM_MODEL, base_url=base_url, temperature=0)

    st.error(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use openai, anthropic, groq, or ollama.")
    st.stop()


llm = get_llm()

with st.sidebar:
    st.subheader("⚙️ LLM Settings")
    st.write(f"**Provider:** {LLM_PROVIDER}")
    st.write(f"**Model:** {LLM_MODEL}")
    st.caption("Change LLM_PROVIDER / LLM_MODEL in your .env file to switch.")

# ---------------------------------------------------------------------------
# 2. PERSISTENT IN-MEMORY DUCKDB CONNECTION (one per app session/run)
# ---------------------------------------------------------------------------
# st.cache_resource keeps ONE connection alive across Streamlit reruns,
# so tables we load don't disappear every time the user interacts with the UI.
@st.cache_resource
def get_duckdb_connection():
    return duckdb.connect(database=":memory:")


con = get_duckdb_connection()

# Track which table names currently exist in this DuckDB connection.
if "tables" not in st.session_state:
    st.session_state.tables = {}  # {table_name: dataframe}


# ---------------------------------------------------------------------------
# 3. HELPER: turn a filename into a safe SQL table name
# ---------------------------------------------------------------------------
def safe_table_name(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    name = "".join(c if c.isalnum() else "_" for c in name).lower()
    if not name or name[0].isdigit():
        name = f"t_{name}"
    return name


# ---------------------------------------------------------------------------
# 4. FILE UPLOAD (MULTIPLE FILES)
# ---------------------------------------------------------------------------
st.subheader("1. Upload your files")
uploaded_files = st.file_uploader(
    "Upload one or more CSV / Excel files",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
)

if uploaded_files:
    for f in uploaded_files:
        ext = os.path.splitext(f.name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            st.warning(f"Skipping unsupported file: {f.name}")
            continue

        table_name = safe_table_name(f.name)

        # Avoid re-reading a file we've already loaded into this session.
        if table_name in st.session_state.tables:
            continue

        try:
            if ext == ".csv":
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
        except Exception as e:
            st.error(f"Could not read {f.name}: {e}")
            continue

        # Basic cleaning: trim string columns, drop fully-empty rows.
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()
        df = df.dropna(how="all")

        # Bulk-register the dataframe as a DuckDB table (no row-by-row insert).
        con.register(table_name, df)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {table_name}")
        st.session_state.tables[table_name] = df

        st.success(f"Loaded '{f.name}' as table `{table_name}` ({len(df)} rows, {len(df.columns)} cols)")

if st.session_state.tables:
    st.subheader("2. Preview your tables")
    tabs = st.tabs(list(st.session_state.tables.keys()))
    for tab, (name, df) in zip(tabs, st.session_state.tables.items()):
        with tab:
            st.dataframe(df.head(100), use_container_width=True)
            st.caption(f"{len(df)} rows x {len(df.columns)} columns")


# ---------------------------------------------------------------------------
# 5. BUILD TOKEN-EFFICIENT SCHEMA CONTEXT FOR ALL TABLES
# ---------------------------------------------------------------------------
def build_schema_context() -> str:
    """
    Builds a compact text block describing every loaded table:
    name, columns + types, row count, and a few sample rows.
    This is the ONLY thing about the data we send to the LLM -
    never the full dataset. This is what makes JOIN generation
    possible: the LLM can see all table names/columns at once
    and pick the right join keys itself.
    """
    blocks = []
    for name, df in st.session_state.tables.items():
        cols = ", ".join(f"{c} ({str(t)})" for c, t in df.dtypes.items())
        sample = df.head(MAX_SAMPLE_ROWS).to_csv(index=False)
        blocks.append(
            f"Table: {name}\n"
            f"Row count: {len(df)}\n"
            f"Columns: {cols}\n"
            f"Sample rows:\n{sample}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 6. SQL GENERATION VIA LANGCHAIN
# ---------------------------------------------------------------------------
SQL_SYSTEM_PROMPT = """
You are an expert DuckDB SQL generator.

You will be given one or more table schemas, row counts, and sample rows.
Generate exactly one valid DuckDB SELECT query that answers the user's question.

Rules:
- Generate ONLY DuckDB-compatible SQL.
- Return ONLY the SQL query. No markdown, explanations, or code fences.
- Only generate SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, COPY, or ATTACH.
- Use only the tables and columns provided in the schema.
- Never invent table names or column names.
- Quote identifiers (table or column names) with double quotes ONLY if they contain spaces or special characters.
- Use single quotes for string literals.
- Use DuckDB SQL syntax (not MySQL, PostgreSQL, SQL Server, or SQLite specific syntax).
- If multiple tables are needed, infer the correct JOIN using matching column names.
- If the answer cannot be determined from the available tables, return:
SELECT 'Unable to answer with available data' AS message;
"""


def generate_sql(question: str, schema_context: str) -> str:
    messages = [
        SystemMessage(content=SQL_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Schemas:\n{schema_context}\n\nQuestion: {question}\n\nSQL:"
        ),
    ]
    response = llm.invoke(messages)
    sql = response.content.strip()
    # Strip accidental markdown code fences if the model adds them anyway.
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


# ---------------------------------------------------------------------------
# 7. SQL VALIDATION (safety check before execution)
# ---------------------------------------------------------------------------
def validate_sql(sql: str) -> tuple[bool, str]:
    forbidden = ("insert", "update", "delete", "drop", "alter", "create", "truncate", "attach", "copy")
    lowered = sql.lower()
    if any(word in lowered for word in forbidden):
        return False, "Query rejected: only SELECT statements are allowed."
    try:
        parsed = sqlglot.parse_one(sql, read="duckdb")
        if parsed.key != "select":
            return False, "Query rejected: only SELECT statements are allowed."
    except Exception as e:
        return False, f"Could not parse SQL: {e}"
    return True, ""


# ---------------------------------------------------------------------------
# 8. AUTO CHART SELECTION
# ---------------------------------------------------------------------------
def auto_chart(df: pd.DataFrame):
    if df.shape[0] == 0 or df.shape[1] < 2:
        return None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    other_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

    try:
        if datetime_cols and numeric_cols:
            return px.line(df, x=datetime_cols[0], y=numeric_cols[0])
        if other_cols and numeric_cols:
            return px.bar(df, x=other_cols[0], y=numeric_cols[0])
        if len(numeric_cols) >= 2:
            return px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
        if len(numeric_cols) == 1:
            return px.histogram(df, x=numeric_cols[0])
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 9. RESULT EXPLANATION VIA LANGCHAIN
# ---------------------------------------------------------------------------
EXPLAIN_SYSTEM_PROMPT = """
You are a business data analyst.

Explain the query result in 1-2 simple sentences.

Rules:
- Do not invent facts or assumptions.
- If the result is empty, state that no matching records were found.
- Do not explain the SQL query.
"""

def explain_result(question: str, result_df: pd.DataFrame) -> str:
    MAX_LLM_ROWS = 10

    # No data
    if result_df.empty:
        return "No matching records were found."

    # Small result set - let the LLM summarize
    result_data = result_df.to_csv(index=False)

    messages = [
        SystemMessage(content=EXPLAIN_SYSTEM_PROMPT),
        HumanMessage(
            content=f"""
                    Question:
                    {question}

                    Rows Returned:
                    {len(result_df)}

                    Columns:
                    {", ".join(result_df.columns)}

                    Result:
                    {result_data}
                    """
        ),
    ]

    return llm.invoke(messages).content.strip()

# ---------------------------------------------------------------------------
# 10. CHAT INTERFACE
# ---------------------------------------------------------------------------
st.subheader("3. Ask a question about your data")

if not st.session_state.tables:
    st.info("Upload at least one file above to start asking questions.")

else:
    question = st.text_input(
        "e.g. 'Join orders and customers, show total spend per customer'"
    )

    if st.button("Ask", type="primary") and question:

        # Build schema context for the LLM
        schema_context = build_schema_context()

        # Generate SQL
        with st.spinner("Generating SQL..."):
            sql = generate_sql(question, schema_context)

        st.code(sql, language="sql")

        # Validate generated SQL
        is_valid, error_msg = validate_sql(sql)

        if not is_valid:
            st.error(error_msg)

        else:
            try:
                # Execute SQL
                start = time.time()
                result_df = con.execute(sql).fetchdf()
                elapsed = time.time() - start

                st.success(f"Returned {len(result_df)} rows in {elapsed:.2f}s")

                # Show complete result
                st.dataframe(result_df, use_container_width=True)

                # Auto chart (if applicable)
                fig = auto_chart(result_df)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

                # Generate explanation only when useful
                if len(result_df) <= 10:
                    with st.spinner("Generating explanation..."):
                        explanation = explain_result(question, result_df)
                else:
                    explanation = (
                        "The complete answer is displayed in the table above. "
                    )

                st.markdown("### Explanation")
                st.write(explanation)

            except Exception as e:
                st.error(f"Error executing SQL: {e}")