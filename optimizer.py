from models import TextModel
import pandas as pd
import random
import re

class PromptOptimizer:
    def __init__(self, model_name="mosaicml/mpt-1b-redpajama-200b-dolly"):
        self.model_wrapper = TextModel(model_name)
        self.device = self.model_wrapper.device

    def score_bcr(self, output_text):
        """
        A very simple proxy for 'BCR' (Balance-Context Relevance).
        Higher score if answer includes financial keywords.
        """
        finance_keywords = ["market", "stock", "investment", "inflation", "returns", "risk", "interest", "economy"]
        matches = sum(1 for word in finance_keywords if word in output_text.lower())
        return round(min(1.0, matches / len(finance_keywords) * 2), 3)

    def identify_root_cause(self, output_text):
        """
        Heuristic-based root cause analysis — very simplified.
        """
        if len(output_text.split()) < 30:
            return "Too brief or lacks explanation"
        elif not any(word in output_text.lower() for word in ["example", "for instance", "e.g."]):
            return "Missing illustrative example"
        elif "investment" not in output_text.lower():
            return "Key finance concept missing"
        else:
            return "Lacks clarity or precision"

    def reflect_on_prompt(self, prompt, root_cause):
        """
        Self-reflection logic — use root cause to generate refined prompt.
        """
        reflection_instruction = f"Refine this financial prompt to fix the issue: {root_cause}."
        full_prompt = f"{prompt} {reflection_instruction}"
        new_prompt = self.model_wrapper.generate(full_prompt, max_new_tokens=60)
        return new_prompt.strip()

    def optimize(self, initial_prompt, num_rounds=1):
        all_data = []
        best_prompt = initial_prompt
        best_score = 0.0

        # Generate LLM output for initial prompt
        initial_output = self.model_wrapper.generate(initial_prompt)
        initial_score = self.score_bcr(initial_output)
        print(f"Initial BCR Score: {initial_score}")

        # Root cause
        root_cause = self.identify_root_cause(initial_output)

        # Generate 10 refined prompts
        refined_prompts = []
        for i in range(10):
            refined = self.reflect_on_prompt(initial_prompt, root_cause)
            refined_prompts.append(refined)

        # Evaluate each refined prompt
        for idx, prompt in enumerate(refined_prompts, start=1):
            output = self.model_wrapper.generate(prompt)
            score = self.score_bcr(output)
            root_fixed = root_cause
            all_data.append({
                "Prompt No.": idx,
                "Prompt": prompt,
                "Output": output[:100].replace("\n", " ") + "...",
                "Root Cause Fixed": root_fixed,
                "BCR Score": score
            })
            if score > best_score:
                best_score = score
                best_prompt = prompt

        df = pd.DataFrame(all_data)
        return best_prompt, best_score, df

