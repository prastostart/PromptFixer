# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from optimizer import PromptOptimizer

app = FastAPI(title="Prompt Optimizer Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str
    num_rounds: int = 2

optimizer = PromptOptimizer()  # small LLM for demo

@app.post("/optimize")
def optimize_prompt(request: PromptRequest):
    user_prompt = request.prompt
    print(f"Received prompt: {user_prompt[:50]}...")
    best_prompt, best_score, df = optimizer.optimize(user_prompt, num_rounds=request.num_rounds)
    print(f"Best prompt: {best_prompt[:50]}..., Best score: {best_score}")
    return {
        "best_prompt": best_prompt,
        "best_score": best_score,
        "all_candidates": df.to_dict(orient="records")
    }
