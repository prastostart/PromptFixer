# main.py

from optimizer import PromptOptimizer
from logger import init_logger

def main():
    # Initialize log file
    init_logger()

    # === Specify the model ===
    model_name = "microsoft/phi-2"

    # Initialize optimizer
    optimizer = PromptOptimizer(model_name=model_name)

    # === Define task and data ===
    initial_prompt = "Classify this text as positive or negative."
    ground_truth = "positive"

    # Run optimization
    best_prompt, best_score = optimizer.optimize(
        initial_prompt=initial_prompt,
        ground_truth=ground_truth,
        num_rounds=5
    )

    # Final summary
    print("\n===== FINAL RESULTS =====")
    print(f"Best Prompt: {best_prompt}")
    print(f"Best Score: {best_score:.3f}")

if __name__ == "__main__":
    main()

