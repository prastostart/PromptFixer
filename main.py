# main.py
from optimizer import PromptOptimizer, plot_bcr_scores
from logger import init_logger
import pandas as pd

def main():
    # Initialize logging
    init_logger()

    model_name = "mosaicml/mpt-1b-redpajama-200b-dolly"
    optimizer = PromptOptimizer(model_name=model_name)

    # ----------- CHANGED -------------
    # 1. Dynamic user input instead of hardcoded prompt
    initial_prompt = input("Enter your prompt (default: 'Explain how inflation affects investment returns.'): ") \
                     or "Explain how inflation affects investment returns."

    # 2. Dynamic number of optimization rounds
    num_rounds = input("Enter number of optimization rounds (default 5): ")
    try:
        num_rounds = int(num_rounds)
    except ValueError:
        num_rounds = 5

    print(f"\n=== Initial Prompt ===\n{initial_prompt}")
    print(f"Running {num_rounds} optimization rounds...\n")
    # ---------------------------------

    # Run the optimizer
    best_prompt, best_score, df = optimizer.optimize(initial_prompt=initial_prompt, num_rounds=num_rounds)

    # Display results
    print("\n=== Generated Prompts and Scores ===")
    print(df.to_string(index=False))

    print("\n===== FINAL RESULTS =====")
    print(f"Best Prompt:\n{best_prompt}")
    print(f"Best Score: {best_score}")

    # Plot scores
    plot_bcr_scores(df)

    # Save CSV logs
    df.to_csv("prompt_logs.csv", index=False)
    with open("best_prompts.csv", "w", encoding="utf-8") as f:
        f.write(f"best_prompt,best_score\n{best_prompt},{best_score}\n")

if __name__ == "__main__":
    main()
