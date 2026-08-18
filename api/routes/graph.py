import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from api.dependencies import agent
from sql_agent.utils.error_sanitizer import sanitize_error
from sql_agent.utils.error_sanitizer import sanitize_error

router = APIRouter(tags=["graph"])
logger = logging.getLogger(__name__)


@router.get("/graph", response_class=HTMLResponse)
async def get_graph():
    """Render the LangGraph workflow as an interactive Mermaid diagram."""
    try:
        graph = agent.get_graph()
        mermaid_code = graph.draw_mermaid()

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LangGraph Workflow Diagram</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: Arial, sans-serif;
            background: #f5f5f5;
            padding: 24px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            background: #fff;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,.1);
            padding: 28px;
        }}
        h1 {{
            color: #222;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .info {{
            background: #e8f5e9;
            border-left: 4px solid #4CAF50;
            padding: 14px 18px;
            border-radius: 0 6px 6px 0;
            margin-bottom: 24px;
        }}
        .info h3 {{ margin-bottom: 8px; color: #2e7d32; }}
        .info ul {{ padding-left: 20px; line-height: 1.8; }}
        .mermaid {{ text-align: center; margin: 24px 0; }}
        .actions {{ text-align: center; margin-top: 20px; display: flex; gap: 12px; justify-content: center; }}
        .btn {{
            padding: 10px 22px;
            border-radius: 6px;
            color: #fff;
            text-decoration: none;
            font-size: 14px;
            font-weight: 600;
        }}
        .btn-green {{ background: #4CAF50; }}
        .btn-blue  {{ background: #2196F3; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SQL Agent Workflow</h1>
        <div class="info">
            <h3>Node Descriptions</h3>
            <ul>
                <li><strong>list_tables</strong> — Lists all available database tables</li>
                <li><strong>call_get_schema</strong> — Decides which table schemas to fetch next</li>
                <li><strong>get_schema</strong> — Retrieves schema + sample rows for selected tables</li>
                <li><strong>schema_analysis</strong> — Decides if enough schema has been gathered or more is needed</li>
                <li><strong>generate_query</strong> — Generates a SQL query from the question</li>
                <li><strong>check_query</strong> — Validates and rewrites the SQL if needed</li>
                <li><strong>run_query</strong> — Executes the SQL and returns results</li>
            </ul>
        </div>
        <div class="mermaid">
{mermaid_code}
        </div>
        <div class="actions">
            <a class="btn btn-green" href="/">← Back to Home</a>
        </div>
    </div>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>"""
        return HTMLResponse(content=html_content)

    except Exception as e:
        safe_message = sanitize_error(e, context="GET /graph")
        raise HTTPException(status_code=500, detail=safe_message)
