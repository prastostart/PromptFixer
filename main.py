from optimizer import PromptOptimizer
from logger import init_logger
import pandas as pd

def main():
    init_logger()

    model_name = "mosaicml/mpt-1b-redpajama-200b-dolly"

    optimizer = PromptOptimizer(model_name=model_name)

    # Generate an initial prompt related to finance
    initial_prompt = "Explain how inflation affects investment returns."
    print(f"\n=== Initial Prompt ===\n{initial_prompt}")

    # Run optimization for ONE round
    best_prompt, best_score, df = optimizer.optimize(
        initial_prompt=initial_prompt,
        num_rounds=1
    )

    # Print the generated table of results
    print("\n=== Generated Prompts and Scores ===")
    print(df.to_string(index=False))

    # Print and save best prompt
    print("\n===== FINAL RESULTS =====")
    print(f"Best Prompt: {best_prompt}")
    print(f"Best Score (BCR): {best_score}")

    df.to_csv("prompt_logs.csv", index=False)
    with open("best_prompts.csv", "w", encoding="utf-8") as f:
        f.write(f"best_prompt,best_score\n{best_prompt},{best_score}\n")

if __name__ == "__main__":
    main()

