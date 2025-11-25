# optimizer.py
from models import TextModel
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import matplotlib.pyplot as plt

class PromptOptimizer:
    def __init__(self, model_name="distilgpt2", retriever=None):
        # Use small model for fast demo
        self.model_wrapper = TextModel(model_name)
        self.device = self.model_wrapper.device
        self.retriever = retriever
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---------------- Embedding & Similarity ----------------
    def embed(self, text: str):
        return self.embedding_model.encode(text, convert_to_tensor=True)

    def cosine_sim(self, a, b):
        return float(util.cos_sim(a, b).item())

    # ---------------- Penalties ----------------
    def drift_penalty(self, prompt, output):
        return 1 - self.cosine_sim(self.embed(prompt), self.embed(output))

    def repetition_penalty(self, text):
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
        if len(sentences) < 2:
            return 0.0
        freq = {}
        for s in sentences:
            freq[s] = freq.get(s, 0) + 1
        probs = np.array(list(freq.values())) / len(sentences)
        entropy = -np.sum(probs * np.log2(probs))
        return 1 - entropy / np.log2(len(sentences))

    def hallucination_penalty(self, prompt, output):
        prompt_tokens = set(prompt.lower().split())
        output_tokens = set(output.lower().split())
        hallucinated = [t for t in output_tokens if t not in prompt_tokens]
        return min(len(hallucinated) * 0.05, 0.5)

    # ---------------- BCR Scoring ----------------
    def compute_bcr(self, ref_output, new_output):
        if not ref_output.strip():
            return 0.5  # neutral score first round
        return round(max(0, min(1, 1 - self.cosine_sim(self.embed(ref_output), self.embed(new_output)))), 4)

    def compute_final_score(self, prompt, ref_output, output):
        base_bcr = self.compute_bcr(ref_output, output)
        drift = self.drift_penalty(prompt, output)
        hall = self.hallucination_penalty(prompt, output)
        rep = self.repetition_penalty(output)
        # Reduced weights for demo
        final = 0.7*base_bcr + 0.3*(1-(0.2*drift + 0.1*hall + 0.1*rep))

        return {
            "final_score": round(max(0, min(1, final)), 4),
            "base_bcr": base_bcr,
            "penalties": {"drift": round(drift,2), "hallucination": round(hall,2), "repetition": round(rep,2)}
        }

    # ---------------- Root Cause ----------------
    def determine_root_cause(self, prompt, ref_output, output):
        penalties = self.compute_final_score(prompt, ref_output, output)["penalties"]
        causes = []
        if penalties["drift"] > 0.3: causes.append("off-topic")
        if penalties["hallucination"] > 0.2: causes.append("unsupported facts")
        if penalties["repetition"] > 0.25: causes.append("repetitive")
        if not causes: causes.append("needs clarity")
        return "; ".join(causes)

    # ---------------- Prompt Refinement ----------------
    def reflect_on_prompt(self, prompt, root_cause):
        # For demo: just append a note
        return f"{prompt} [refined to fix: {root_cause}]"

    # ---------------- Optimization Loop ----------------
    def optimize(self, initial_prompt, num_rounds=2, num_candidates=3):
        all_data = []
        best_prompt = initial_prompt
        ref_output = self.model_wrapper.generate(initial_prompt, max_new_tokens=50, temperature=0.3)
        best_score = 0.0

        for rnd in range(1, num_rounds+1):
            print(f"=== Round {rnd} ===")
            root_cause = self.determine_root_cause(best_prompt, ref_output, ref_output)
            candidates = [self.reflect_on_prompt(best_prompt, root_cause)+f" [{i}]" for i in range(1,num_candidates+1)]

            for idx, prompt in enumerate(candidates,1):
                output = self.model_wrapper.generate(prompt, max_new_tokens=50, temperature=0.3)
                score_data = self.compute_final_score(prompt, ref_output, output)
                score = score_data["final_score"]
                print(f"Candidate {idx}: score={score}, prompt='{prompt[:30]}...'")
                all_data.append({
                    "Round": rnd,
                    "Prompt No.": idx,
                    "Prompt": prompt,
                    "Output": output[:100].replace("\n"," ")+"...",
                    "Final Score": score,
                    "Penalties": score_data["penalties"]
                })
                if score > best_score:
                    best_score = score
                    best_prompt = prompt
                    ref_output = output

        return best_prompt, best_score, pd.DataFrame(all_data)

# ---------------- Plot ----------------
def plot_bcr_scores(df):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    for r in df["Round"].unique():
        subset = df[df["Round"]==r]
        plt.plot(subset["Prompt No."], subset["Final Score"], marker="o", label=f"Round {r}")
    plt.xlabel("Candidate Prompt No.")
    plt.ylabel("Score")
    plt.title("BCR Scores Across Rounds")
    plt.ylim(0,1)
    plt.legend()
    plt.grid(True)
    plt.show()
