# SQL Agent

An agentic Natural-Language-to-SQL system powered by LangGraph, FastAPI, and structured database knowledge layers.

Ask complex analytical questions in plain English and get verified, schema-aware SQL queries, execution traces, and structured tabular results.

---

## Overview

SQL Agent bridges the gap between natural language questions and enterprise data warehouses using an iterative, multi-step agentic workflow:

- **Schema Discovery & Pruning**: Iteratively identifies and retrieves relevant table schemas and foreign key relationships without overflowing context limits.
- **Three-Tier Knowledge Architecture**: Combines automatic schema relationship mapping (foreign keys / surrogate keys), live low-cardinality column value profiling, and verified few-shot query patterns.
- **Strict Read-Only Safety**: Enforces read-only guarantees at both the driver and engine levels (blocking DML, DDL, and dynamic SQL execution).
- **Multi-Backend Portability**: Built on SQLAlchemy and LangChain abstractions supporting SQLite, Azure Synapse / SQL Server, and DuckDB (for analytical benchmarks like TPC-DS).
- **Execution & Observability**: Visualizes step-by-step execution traces, captures structured tabular data for UI rendering, and renders the LangGraph topology in Mermaid.

---

## Tech Stack

| Component | Technology |
| --- | --- |
| **LLM & Reasoning** | Azure OpenAI (GPT-4o) / OpenAI compatible |
| **Agent Orchestration** | LangGraph StateGraph |
| **API Framework** | FastAPI + Uvicorn |
| **Database Backends** | SQLite, Azure Synapse / SQL Server, DuckDB |
| **Data Access & Safety** | SQLAlchemy, LangChain SQLDatabase, pyodbc |
| **Frontend** | Lightweight Single-Page Interface (Vanilla HTML/JS) |

---

## Project Structure

\\\
sql_agent/
+-- api/
¦   +-- routes/          # FastAPI endpoints (/query, /health, /graph, /database)
¦   +-- app.py           # Application factory
¦   +-- dependencies.py  # Agent and DB lifecycle initialization
¦   +-- schemas.py       # Pydantic request/response models
+-- sql_agent/
¦   +-- config/          # Settings & structured logging
¦   +-- database/        # Multi-backend connection manager & write-guard
¦   +-- graph/           # LangGraph orchestration state machine
¦   +-- knowledge/       # Auto-profiler, relationship discovery & example loader
¦   +-- nodes/           # Graph nodes (list_tables, schema_analysis, query generation, validator, runner)
¦   +-- prompts/         # Governed prompt templates & rule integration
¦   +-- tools/           # LangChain SQL database toolkit bindings
¦   +-- utils/           # Error sanitization & helper utilities
+-- frontend/
¦   +-- index.html       # Web UI with execution trace & data table visualization
+-- profiles/            # Deployment & benchmark profiles (rules, examples, config)
+-- requirements.txt     # Python dependencies
+-- main.py              # Application entrypoint
\\\

---

## Quickstart

### 1. Environment Setup

\\\ash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt
\\\

### 2. Configuration

Copy the sample environment file:

\\\ash
cp .env.example .env
\\\

Configure your environment variables in \.env\:
- \AZURE_OPENAI_API_KEY\, \AZURE_OPENAI_ENDPOINT\, \AZURE_OPENAI_DEPLOYMENT\
- \DB_BACKEND\ (\sqlite\ or \synapse\)
- Database connection details (if using Synapse)

### 3. Run the Server

\\\ash
uvicorn main:app --port 8000 --reload
\\\

Access the UI at \http://localhost:8000\ and the interactive API documentation at \http://localhost:8000/docs\.

---

## Workflow Architecture

The agent executes as a LangGraph state machine with cyclic refinement:

\\\
[START]
   ¦
   ?
[list_tables] --? [call_get_schema] --? [get_schema] --? [schema_analysis]
                                                               ¦
                                       +-----------------------------------------------+
                                       ?                                               ?
                              (NEED_MORE_SCHEMAS)                             (ANALYSIS_COMPLETE)
                              Loop back to call_get_schema                             ¦
                              (up to MAX_SCHEMA_ITERATIONS)                            ?
                                                                               [generate_query]
                                                                                       ¦
                                                                                       ?
                                                                                 [check_query]
                                                                                       ¦
                                                                                       ?
                                                                                  [run_query]
                                                                                       ¦
                                                                                       ?
                                                                               [generate_query]
                                                                               (Final NL synthesis)
                                                                                       ¦
                                                                                       ?
                                                                                     [END]
\\\

### Pipeline Nodes

1. **\list_tables\**: Discovers available tables in the connected database catalog.
2. **\call_get_schema\**: Heuristically selects candidate tables based on the user query.
3. **\get_schema\**: Fetches detailed DDL, columns, and sample rows.
4. **\schema_analysis\**: Structured output evaluation checking if all required dimension and fact tables are present in context; requests additional schemas if missing.
5. **\generate_query\**: Produces a syntactically correct dialect-specific SQL query incorporating business rules and column value profiling.
6. **\check_query\**: Semantic and syntax validation pass preventing common LLM SQL errors, missing filters, and invalid join clauses.
7. **\un_query\**: Safely executes the validated SQL query, capturing both string results for the LLM and structured row/column matrices for the UI.

---

## Knowledge Layers

1. **Layer 1 (Schema & Topology Discovery)**: Automatic join-path inference based on key/identifier patterns across tables.
2. **Layer 2 (Categorical Profiling)**: Automatic extraction of distinct values for low-cardinality columns (e.g. status codes, country names, segments) to prevent LLM hallucinations on filter literals.
3. **Layer 3 (Governed Examples & Rules)**: Verified few-shot NL-to-SQL exemplars and modular business rule definitions.

---

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| \GET\ | \/\ | Serves the web execution interface |
| \POST\ | \/query\ | Executes natural-language query and returns answer, trace, and table data |
| \GET\ | \/health\ | Healthcheck and active database status |
| \GET\ | \/graph\ | Renders interactive Mermaid workflow diagram |
| \GET\ | \/database/tables\ | Lists accessible database tables |
| \GET\ | \/database/schema/{table}\ | Inspects table schema and sample records |

---

## Security & Safety

- **Read-Only Enforcement**: Unconditional driver-level read-only mode for SQLite (\mode=ro\) and event-level execution interception blocking write operations (\INSERT\, \UPDATE\, \DELETE\, \DROP\, \ALTER\, \TRUNCATE\, \EXEC\).
- **Error Sanitization**: Sanitized error handler mapping database exceptions to safe user-facing error categories while logging correlation IDs for debugging.
