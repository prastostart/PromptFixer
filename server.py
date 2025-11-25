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
    num_rounds: int = 3

optimizer = PromptOptimizer(model_name="meta-llama/Llama-3.2-1B-Instruct")

@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "index.html not found"}

@app.post("/optimize")
def optimize_endpoint(request: PromptRequest):
    def iter_response():
        # Fully automated loop
        for step_data in optimizer.optimize(request.prompt, num_rounds=request.num_rounds):
            yield json.dumps(step_data) + "\n"

    return StreamingResponse(iter_response(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)