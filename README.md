# AI CSV Chat

AI CSV Chat is a lightweight GenAI application that allows users to upload CSV or Excel files and query their data using natural language. Instead of sending the entire dataset to an LLM, the application stores the data in an embedded DuckDB database, generates SQL using an LLM, executes the query locally, and returns the results. This approach reduces token usage while providing fast and accurate answers.

## Features

* Upload CSV and Excel files
* Natural language to SQL using an LLM
* Embedded DuckDB database (no setup required)
* Secure SQL validation (SELECT queries only)
* Automatic charts for supported results
* AI-generated explanations for small result sets
* Low token usage by executing queries locally

## Tech Stack

* Python
* Streamlit
* DuckDB
* Pandas
* Plotly
* LangChain
* SQLGlot
* uv

## How It Works


```text
               Upload CSV / Excel
                        │
                        ▼
                  Pandas DataFrame
                        │
                        ▼
            Data Cleaning & Validation
                        │
                        ▼
              Embedded DuckDB Database
                        │
                        ▼
          Build Schema + Sample Context
                        │
                        ▼
                  LLM Generates SQL
                        │
                        ▼
                 SQL Validation Layer
                        │
                        ▼
               Execute Query in DuckDB
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
     Complete Result          Auto Visualization
            │
            ▼
   Small Result (≤10 rows)?
            │
      ┌─────┴─────┐
      │           │
     Yes         No
      │           │
      ▼           ▼
 LLM Explanation  Show Result Only
```

## Run Locally

```bash
git clone <repository-url>
cd ai-csv-chat

uv sync

uv run streamlit run app.py
```

Create a `.env` file before running:

```env
# Choose provider: openai | anthropic | groq | ollama
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Only fill in the key for the provider you selected above.
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
GROQ_API_KEY=your-groq-key-here

# Only needed if LLM_PROVIDER=ollama (local models, no API key required)
OLLAMA_BASE_URL=http://localhost:11434
```

## Why This Project?

This project demonstrates a cost-efficient GenAI architecture by using an LLM only for SQL generation and concise explanations, while all data processing and analytics are performed locally in DuckDB.
