from optimizer import PromptOptimizer
# from logger import init_logger # Uncomment if you actually have a logger.py

import pandas as pd

def main():
    # init_logger() 
    
    # CHANGE 1: No need to pass model_name, TextModel handles Qwen GGUF internally
    optimizer = PromptOptimizer()
    
    initial_prompt = input("Enter prompt: ") or "Explain quantum physics."
    
    # Note: Rounds are currently hardcoded to 5 in optimizer.py. 
    # To change this, you would need to update the optimize() method definition.
    print(f"\n--- Starting Diagnostic Optimization ---")
    
    best_prompt = ""
    best_score = 0
    df = pd.DataFrame()

    # CHANGE 2: Removed 'num_rounds=rounds' because your optimize() method 
    # defined in the first prompt does not accept arguments other than initial_prompt.
    for step_data in optimizer.optimize(initial_prompt):
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