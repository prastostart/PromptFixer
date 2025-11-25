from optimizer import PromptOptimizer
from logger import init_logger
import pandas as pd

def main():
    init_logger()
    optimizer = PromptOptimizer(model_name="meta-llama/Llama-3.2-1B-Instruct")
    
    initial_prompt = input("Enter prompt: ") or "Explain quantum physics."
    rounds = 3
    
    print(f"\n--- Starting Diagnostic Optimization ({rounds} rounds) ---")
    
    best_prompt = ""
    best_score = 0
    df = pd.DataFrame()

    for step_data in optimizer.optimize(initial_prompt, num_rounds=rounds):
        if step_data["type"] == "progress":
            print(f"[AGENT] {step_data['message']}")
        elif step_data["type"] == "result":
            best_prompt = step_data["best_prompt"]
            best_score = step_data["best_score"]
            df = pd.DataFrame(step_data["all_candidates"])

    print("\n=== FINAL DIAGNOSIS & PRESCRIPTION ===")
    print(f"Best Prompt Found (Score {best_score:.4f}):")
    print(best_prompt)

if __name__ == "__main__":
    main()