from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    question: str


class TableData(BaseModel):      # ← NEW: structured table payload for the UI
    columns: List[str]
    rows: List[List[Any]]


class QueryResponse(BaseModel):
    question: str
    messages: List[Dict[str, Any]]
    final_answer: str
    table_data: Optional[TableData] = None   # ← NEW: None when no rows were returned
    node_timings: Optional[Dict[str, float]] = None  # {node_name: elapsed_seconds}
