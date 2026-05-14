from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from agent import run_agent
import uvicorn

app = FastAPI(title="Agentic RAG API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = []
    summary: Optional[str] = ""

def serialize_message(msg: BaseMessage) -> Dict[str, str]:
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    elif isinstance(msg, AIMessage):
        return {"role": "assistant", "content": msg.content}
    return {"role": "system", "content": msg.content}

def deserialize_history(history: List[Dict[str, str]]) -> List[BaseMessage]:
    messages = []
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))
    return messages

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        history = deserialize_history(request.history)
        result = run_agent(request.query, history=history, summary=request.summary)
        
        # Extract the latest answer and the thought process
        final_answer = result["messages"][-1].content
        thought_process = result.get("thought_process", [])
        next_step = result.get("next_step", "")
        updated_summary = result.get("summary", "")
        
        # Prepare the serializable history
        new_history = [serialize_message(m) for m in result["messages"]]
        
        return {
            "answer": final_answer,
            "thought_process": thought_process,
            "next_step": next_step,
            "summary": updated_summary,
            "history": new_history
        }
    except Exception as e:
        print(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    return {"status": "online", "model": "llama3.2", "embeddings": "nomic-embed-text"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
