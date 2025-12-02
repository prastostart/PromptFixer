from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from optimizer import PromptOptimizer
import uvicorn
import os
import json
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str
    use_rag: bool = False

optimizer = PromptOptimizer()

@app.get("/")
def read_root():
    return FileResponse("index.html") if os.path.exists("index.html") else {"error": "index.html missing"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Save temp file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Read and Ingest
    with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    # Add to Retriever
    num_chunks = optimizer.retriever.add_document(text, filename=file.filename)
    
    os.remove(temp_path)
    return {"message": f"Ingested {file.filename} into {num_chunks} chunks."}

@app.post("/reset_db")
def reset_db():
    optimizer.retriever.clear_db()
    return {"message": "Knowledge base cleared."}

@app.post("/optimize")
def optimize_endpoint(request: PromptRequest):
    def iter_response():
        # Pass the RAG flag to the optimizer
        for step_data in optimizer.optimize(request.prompt, use_rag=request.use_rag):
            yield json.dumps(step_data) + "\n"

    return StreamingResponse(iter_response(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)