from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from database import get_current_dialect
from agent import app_graph

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Local SQL Agent", 
    description="LangGraph + Ollama + FastAPI SQL Agent"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class QueryResponse(BaseModel):
    final_answer: str
    sql_query: str | None = None
    sql_result: str | None = None

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        initial_state = {
            "query": request.query,
            "dialect": get_current_dialect(),
            "error_count": 0,
            "chat_history": []
        }
        
        # app_graph.ainvoke çalıştırarak graph akışını tamamen asenkron başlatıyoruz.
        config = {"configurable": {"thread_id": request.session_id}}
        final_state = await app_graph.ainvoke(initial_state, config)
        
        return QueryResponse(
            final_answer=final_state.get("final_answer", ""),
            sql_query=final_state.get("sql_query"),
            sql_result=final_state.get("sql_result")
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)