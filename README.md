# SQL Agent

<div align="center">

**Enterprise Natural-Language-to-SQL Agent with Governed Semantic Layers & Dynamic Schema Intelligence**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![DuckDB](https://img.shields.io/badge/engine-DuckDB-FFF000.svg)](https://duckdb.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Architecture](#system-architecture) •
[Knowledge Engine](#three-tier-knowledge-engine) •
[Semantic Rule Governance](#profile-based-semantic-governance) •
[Quickstart](#quickstart) •
[API Reference](#api-reference) •
[Security](#security--defense-in-depth)

</div>

---

## Executive Overview

**SQL Agent** is an agentic Natural-Language-to-SQL (NL-to-SQL) system engineered to translate complex analytical questions into verified, dialect-optimized SQL over enterprise data warehouses and analytical databases.

Standard NL-to-SQL architectures struggle with four fundamental failure modes:
1. **Context Window Saturation**: Large enterprise schemas (>20 tables, hundreds of columns) overwhelm LLM context windows if loaded statically.
2. **Filter Literal Hallucinations**: Models guess filter values (e.g., `'Completed'` vs `'Complete'`, `'USA'` vs `'US'`) without database context.
3. **Metric & Business Rule Drift**: Complex business definitions (e.g., closing rate formulas, active record flags, composite statuses) cannot be inferred from raw column names alone.
4. **SQL Vulnerabilities & DML Risks**: Accidental or adversarial write operations on production data layers.

SQL Agent addresses these challenges through a **cyclic LangGraph state machine**, a **three-tier startup knowledge discovery engine**, **profile-based semantic rule governance**, and **multi-tier driver/engine security guards**.

---

## Core Capabilities

- 🔄 **Cyclic Schema Discovery Loop**: Dynamically navigates database catalogs using structured Pydantic decision-making (`ANALYSIS_COMPLETE` vs `NEED_MORE_SCHEMAS`) to iteratively pull exact schema definitions without context overflow.
- 🧠 **Three-Tier Knowledge Engine**: Auto-discovers cross-table topological relationships, samples and caches low-cardinality categorical values, and incorporates few-shot query exemplars.
- 📐 **Profile-Based Semantic Governance**: Decouples business rules and KPI definitions from hardcoded prompt text into modular profiles (`profiles/<name>/`), enabling zero-contamination transitions between client deployments and research benchmarks.
- ⚡ **Multi-Backend Portability**: Native abstraction layer supporting **DuckDB** (for analytical research on TPC-DS), **SQLite** (for local development), and **Azure Synapse / SQL Server** (for enterprise data warehouses).
- 🛡️ **Defense-in-Depth Safety**: Enforces read-only access at the database driver level (`mode=ro`, `read_only=True`) and at the SQLAlchemy event listener level, blocking DML, DDL, stored procedure execution, and dynamic SQL.
- 📊 **Real-Time Observability & UI**: Full execution tracing with per-node latency telemetry (`@timed()`), generated SQL syntax inspection, and interactive tabular result rendering.

---

## System Architecture

The orchestration layer is constructed as a cyclic state graph using **LangGraph**:

```
                              [START]
                                 │
                                 ▼
                          [list_tables]
                                 │
                                 ▼
                        [call_get_schema]
                                 │
                                 ▼
                           [get_schema]
                                 │
                                 ▼
                         [schema_analysis] ◄─────────────────────────┐
                                 │                                   │
              ┌──────────────────┴──────────────────┐                │
              ▼                                     ▼                │
     (NEED_MORE_SCHEMAS)                   (ANALYSIS_COMPLETE)       │
    Loop back to fetch more                         │                │
    (up to MAX_ITERATIONS)                          ▼                │
                                             [generate_query]        │
                                                    │                │
                                 ┌──────────────────┴────────┐       │
                                 ▼                           ▼       │
                          (Standard Flow)          (Missing Schema)  │
                                 │                  Call get_schema ─┘
                                 ▼
                           [check_query]
                       (Syntax & Rule Audit)
                                 │
                                 ▼
                            [run_query]
                        (Safe DB Execution)
                                 │
                                 ▼
                          [generate_query]
                        (Final NL Synthesis)
                                 │
                                 ▼
                               [END]
```

### Pipeline Node Breakdown

| Stage | Node Name | Description |
| :--- | :--- | :--- |
| **1. Catalog Discovery** | `list_tables` | Queries catalog to retrieve all accessible tables in the active schema. |
| **2. Heuristic Pruning** | `call_get_schema` | Selects high-probability candidate tables matching user query entities. |
| **3. Schema Hydration** | `get_schema` | Retrieves precise DDL, column types, and sample data rows. |
| **4. Structural Gate** | `schema_analysis` | Evaluates schema sufficiency via structured JSON output (`decision`, `tables_needed`, `reasoning`). Loops back if related dimension/fact tables are missing. |
| **5. Query Generation** | `generate_query` | Compiles dialect-specific SQL leveraging injected semantic rules and categorical profiles. |
| **6. Semantic Validator** | `check_query` | Audits generated SQL for common pitfalls (NULL handling, BETWEEN on datetimes, required metric joins, active record filters). Rewrites if defects exist. |
| **7. Safe Execution** | `run_query` | Executes query against the read-only database backend and formats raw output matrices for UI tabular rendering. |
| **8. Natural Synthesis** | `generate_query` *(Pass 2)* | Transforms query results into an executive natural-language answer with full citation of tables and SQL used. |

---

## Three-Tier Knowledge Engine

At startup, SQL Agent executes automated discovery passes (`sql_agent/knowledge/db_analyzer.py`) to build contextual metadata injected into the agent runtime:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          KNOWLEDGE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 1: Topological Relationship Discovery                            │
│  • Cross-table foreign key & surrogate key inference (*_sk, *_bk)       │
│  • Automated join-path candidate mapping                                │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 2: Categorical Column Profiler                                   │
│  • Live profiling of low-cardinality string columns (2-30 distinct)     │
│  • Caches actual database literals (e.g. status codes, types, regions)  │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 3: Governed Semantic Rules & Exemplars                           │
│  • Domain-specific calculation logic (e.g., conversion, closing rates)   │
│  • Verified NL→SQL few-shot exemplars loaded per profile                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Profile-Based Semantic Governance

To prevent business logic from polluting prompt templates, SQL Agent decouples all domain knowledge into the **Profile System** (`profiles/<profile_name>/`):

```
profiles/
├── client/                     # (Gitignored) Enterprise production rules & config
│   ├── config.yaml             # Backend: Synapse, table prefixes, schemas
│   ├── rules.yaml              # KPI formulas, business logic, global filters
│   └── query_examples.json     # Verified few-shot domain queries
└── tpcds/                      # Public research & benchmark profile
    ├── config.yaml             # Backend: DuckDB, path: tpcds_sf1.duckdb
    ├── rules.yaml              # Synthetic governed benchmark rules
    └── query_examples.json     # Standardized TPC-DS exemplars
```

### Profile Switching

Setting `AGENT_PROFILE` dynamically binds the entire agent stack — swapping database backends, table routing policies, business rules, and few-shot exemplars seamlessly:

```bash
# Run against the local TPC-DS benchmark (DuckDB)
export AGENT_PROFILE=tpcds

# Run against enterprise warehouse deployment
export AGENT_PROFILE=client
```

---

## Tech Stack

| Domain | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) / [LangChain](https://github.com/langchain-ai/langchain) | Cyclic state graph execution, state transitions, and tool bindings |
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | High-performance asynchronous API service & OpenAPI specification |
| **Analytical Engine** | [DuckDB](https://duckdb.org/) | Local columnar database running TPC-DS benchmark scale factor 1 (SF=1) |
| **Enterprise Warehouse**| [Azure Synapse Analytics](https://azure.microsoft.com/en-us/products/synapse-analytics) / MS SQL | Production enterprise data warehousing via `pyodbc` |
| **ORM & Data Layer** | [SQLAlchemy 2.0](https://www.sqlalchemy.org/) | Multi-dialect database abstraction and event-level execution interception |
| **Configuration** | [Pydantic v2](https://docs.pydantic.dev/) + [PyYAML](https://pyyaml.org/) | Validated structured settings and profile configurations |
| **Frontend** | Vanilla JavaScript / Modern CSS | Zero-dependency real-time execution trace & data table UI |

---

## Repository Structure

```
sql_agent/
├── api/
│   ├── routes/
│   │   ├── database.py         # Table discovery & schema inspection endpoints
│   │   ├── graph.py            # Mermaid diagram generation endpoint
│   │   ├── health.py           # System & database connectivity healthcheck
│   │   └── query.py            # Main NL-to-SQL execution endpoint
│   ├── app.py                  # FastAPI application factory
│   ├── dependencies.py         # Agent & DB singleton lifecycle management
│   └── schemas.py              # Pydantic request / response schemas
├── frontend/
│   └── index.html              # Interactive execution trace & table visualizer
├── profiles/
│   ├── tpcds/                  # TPC-DS benchmark profile configuration
│   └── client/                 # (Gitignored) Production rules & credentials
├── sql_agent/
│   ├── config/
│   │   ├── logging_config.py   # Structured logging configuration
│   │   ├── profile_loader.py   # Dynamic profile & rule parser
│   │   └── settings.py         # Central environment & DB settings
│   ├── database/
│   │   └── db.py               # Multi-backend connector & defense-in-depth write blocker
│   ├── graph/
│   │   └── agent.py            # LangGraph StateGraph state machine compilation
│   ├── knowledge/
│   │   ├── db_analyzer.py      # Automated FK mapping & categorical column profiler
│   │   └── knowledge_loader.py # Profile-aware few-shot exemplar retriever
│   ├── nodes/
│   │   ├── list_tables.py      # Catalog discovery node
│   │   ├── query_nodes.py      # SQL generator, validator, and runner nodes
│   │   └── schema_node.py      # Schema retrieval & sufficiency analysis nodes
│   ├── prompts/
│   │   └── sql_prompts.py      # Generic dialect prompt templates with dynamic rule injection
│   ├── tools/
│   │   └── sql_tools.py        # LangChain SQL database tool bindings
│   └── utils/
│       ├── error_sanitizer.py  # User-facing database error sanitizer
│       └── run_agent.py        # Standalone CLI agent runner
├── .env.example                # Sample environment configuration template
├── .gitignore                  # Git tracking exclusions
├── main.py                     # Primary API server entrypoint
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Quickstart

### 1. Prerequisites & Environment Setup

- Python 3.11 or higher
- Git

```bash
# Clone the repository
git clone https://github.com/Ayush4Gupta/sql_agent.git
cd sql_agent

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# or: .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Configure your LLM provider credentials in `.env`:

```ini
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Agent Profile (tpcds | client)
AGENT_PROFILE=tpcds

# Database Backend (duckdb | sqlite | synapse)
DB_BACKEND=duckdb
```

### 3. Generate Benchmark Data (TPC-DS SF=1)

Generate the full 24-table TPC-DS dataset (~315 MB, millions of rows) locally using DuckDB:

```bash
python -c "import duckdb; conn = duckdb.connect('sql_agent/database/tpcds_sf1.duckdb'); conn.execute('INSTALL tpcds; LOAD tpcds;'); conn.execute('CALL dsdgen(sf=1);'); print('Generated TPC-DS SF=1 successfully!'); conn.close()"
```

### 4. Start the Application

Launch the FastAPI application:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive UI**: [http://localhost:8000](http://localhost:8000)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Live Workflow Graph (Mermaid)**: [http://localhost:8000/graph](http://localhost:8000/graph)

---

## API Reference

### `POST /query`
Executes natural-language question translation, execution, and synthesis.

**Request Body:**
```json
{
  "question": "What were the top 5 store sales items by net revenue last quarter?"
}
```

**Response Body:**
```json
{
  "answer": "The top 5 items by net revenue for the previous quarter were generated successfully.",
  "sql_query": "SELECT i.i_item_id, SUM(ss.ss_net_paid) AS net_revenue FROM store_sales ss JOIN item i ON ss.ss_item_sk = i.i_item_sk GROUP BY i.i_item_id ORDER BY net_revenue DESC LIMIT 5",
  "tables_used": ["store_sales", "item"],
  "trace": [
    {
      "node": "list_tables",
      "duration_ms": 12.4,
      "details": { "table_count": 24 }
    },
    {
      "node": "schema_analysis",
      "duration_ms": 341.2,
      "details": { "decision": "ANALYSIS_COMPLETE", "tables": ["store_sales", "item"] }
    }
  ],
  "result_table": {
    "columns": ["i_item_id", "net_revenue"],
    "rows": [
      ["AAAAAAAAEAAAAAAA", 148204.50],
      ["AAAAAAAABAAAAAAA", 123910.20]
    ],
    "row_count": 2
  }
}
```

### Additional Endpoints

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/health` | `GET` | System health check and database connection verification |
| `/graph` | `GET` | Visualizes the active LangGraph topology in Mermaid format |
| `/database/tables` | `GET` | Returns list of accessible database tables |
| `/database/schema/{table_name}` | `GET` | Fetches DDL and column schema for a specified table |

---

## Security & Defense-in-Depth

SQL Agent enforces strict read-only execution guarantees across all connected data warehouses:

1. **Driver-Level Read-Only Guarantees**:
   - **DuckDB**: Connected via `connect_args={"read_only": True}`.
   - **SQLite**: Enforced with URI flag `?mode=ro&uri=true`.
2. **Engine-Level Write Interception**:
   - A SQLAlchemy event listener (`before_cursor_execute`) strips comments and regex-audits all incoming SQL statements against a comprehensive DML/DDL pattern list (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`, `TRUNCATE`, `MERGE`, `EXEC`, `sp_executesql`, etc.).
   - Offending queries raise a `PermissionError` before reaching the database cursor.
3. **Error Sanitization Layer**:
   - Internal database exceptions, stack traces, and connection strings are intercepted by `sanitize_error()`, returning uniform, safe error descriptions to clients while logging detailed correlation IDs internally.

---

## Research Context & Roadmap

This repository forms the core experimental implementation for ongoing research into **governed rule resolution and semantic layers in NL-to-SQL systems**.

- [x] **Track A (Database Migration & Benchmarking)**: TPC-DS SF=1 dataset generation and multi-backend DuckDB integration.
- [ ] **Track B (MCP Semantic Layer Integration)**: Integrating an external Model Context Protocol (MCP) server running a BFS-based rule dependency engine (`resolver.py`) to dynamically resolve composite metrics and versioned business definitions.
- [ ] **Track C (Empirical Evaluation Harness)**: Three-condition comparative evaluation on TPC-DS measuring:
  1. *Baseline* (No business rules)
  2. *Flat Prompt Rules* (Current profile baseline)
  3. *Dynamic Rule Graph* (MCP-resolved rules)
- [ ] **Track D (BIRD Benchmark Integration)**: Generalization verification across multi-domain relational schemas.

---

## License

This project is licensed under the [MIT License](LICENSE).
