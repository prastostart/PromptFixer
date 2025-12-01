from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from optimizer import PromptOptimizer
import uvicorn
import os
import json

app = FastAPI(title="Automated RCA Optimizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

# CHANGE: Initialize without arguments. 
# The new models.py automatically loads the Qwen GGUF model.
optimizer = PromptOptimizer()

@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "index.html not found"}

@app.post("/optimize")
def optimize_endpoint(request: PromptRequest):
    def iter_response():
        # This matches the signature in optimizer.py
        for step_data in optimizer.optimize(request.prompt):
            yield json.dumps(step_data) + "\n"

    return StreamingResponse(iter_response(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)