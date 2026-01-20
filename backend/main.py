import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from dotenv import load_dotenv

from rag_engine import RAGService

load_dotenv()

app = FastAPI(title="RAG Chat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RAGService()

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

@app.get("/")
def read_root():
    return {"status": "ok", "message": "RAG Backend is active"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            result_message = rag_service.ingest_file(temp_path, file.filename)
            status = "success"
        except Exception as ingest_error:
            result_message = f"Ingestion failed: {str(ingest_error)}"
            status = "error"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if status == "error":
             raise HTTPException(status_code=500, detail=result_message)
             
        return {"status": "success", "message": result_message}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        result = rag_service.query(request.message, request.history)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
def reset_db():
    result = rag_service.clear_memory()
    return {"status": "success", "message": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
