from models import TextModel
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re

class PromptOptimizer:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        self.model_wrapper = TextModel(model_name)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.strategy_map = {
            "Off-Topic": "Constraints_Injection",
            "Repetitive": "Anti_Loop_Refinement",
            "Vague": "CO_STAR_Framework",     
            "Illogical": "Explicit_CoT",       
            "Missing Details": "Synthetic_Few_Shot", 
            "Wrong Tone": "Persona_Adoption"
        }

    # ---------------- Metrics ----------------
    def embed(self, text: str):
        return self.embedding_model.encode(text, convert_to_tensor=True)

    def cosine_sim(self, a, b):
        return float(util.cos_sim(a, b).item())

    def drift_penalty(self, prompt, output):
        return 1 - self.cosine_sim(self.embed(prompt), self.embed(output))

    def repetition_penalty(self, text):
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
        if len(sentences) < 2: return 0.0
        freq = {}
        for s in sentences: freq[s] = freq.get(s, 0) + 1
        probs = np.array(list(freq.values())) / len(sentences)
        entropy = -np.sum(probs * np.log2(probs))
        norm_entropy = entropy / np.log2(len(sentences)) if len(sentences) > 1 else 1
        return 1 - norm_entropy

    # ---------------- RCA Module ----------------
    def diagnose_root_cause(self, prompt, output, stats):
        if stats["drift"] > 0.4: return "Off-Topic"
        if stats["repetition"] > 0.3: return "Repetitive"
        if len(output.split()) < 30: return "Missing Details"

        system_prompt = "You are a harsh critic. Identify the fatal flaw in the Output."
        user_query = (
            f"Prompt: {prompt}\nOutput: {output}\n\n"
            "Choose ONE flaw: [Vague, Illogical, Wrong Tone]. "
            "Return ONLY the word."
        )
        try:
            diagnosis = self.model_wrapper.generate(
                prompt=user_query, system_instruction=system_prompt, 
                max_new_tokens=10, temperature=0.1
            ).strip().replace(".", "")
            valid = ["Vague", "Illogical", "Wrong Tone"]
            if any(x in diagnosis for x in valid): return diagnosis
            return "Vague"
        except:
            return "Vague"

    def compute_final_score(self, prompt, output, initial_bad_output):
        if not initial_bad_output: dist = 0.5
        else: dist = 1 - self.cosine_sim(self.embed(initial_bad_output), self.embed(output))

        drift = self.drift_penalty(prompt, output)
        rep = self.repetition_penalty(output)
        
        quality_score = 1 - (0.3*drift + 0.5*rep) 
        final = (0.4 * dist) + (0.6 * quality_score)
        
        return {
            "final_score": round(max(0, min(1, final)), 4),
            "penalties": {"drift": round(drift,2), "repetition": round(rep,2)}
        }

    # ---------------- Strategies ----------------
    def apply_strategy(self, current_prompt, root_cause, strategy):
        candidates = []
        
        base_instruction = ""
        if strategy == "CO_STAR_Framework":
            base_instruction = "Rewrite using CO-STAR format (Context, Objective, Style, Audience, Response)."
        elif strategy == "Explicit_CoT":
            base_instruction = "Rewrite adding strict 'Step-by-Step' reasoning requirements."
        elif strategy == "Synthetic_Few_Shot":
            base_instruction = "Generate 2 relevant examples and append them to the prompt."
        elif strategy == "Persona_Adoption":
            base_instruction = "Adopt a specific expert persona and rewrite the prompt from that perspective."
        elif strategy == "Constraints_Injection":
            base_instruction = "Add a list of 'Negative Constraints' (what NOT to do)."
        else:
            base_instruction = "Rewrite to be more concise and authoritative."

        flavors = [
            "Flavor 1: Extremely Concise and Strict.",
            "Flavor 2: Detailed, Descriptive, and Expansive.",
            "Flavor 3: Unconventional, Creative, and 'Out of the Box'."
        ]

        for flavor in flavors:
            full_instruction = f"{base_instruction}\n\nConstraint: {flavor}\nMake this version distinct."
            new_prompt = self.model_wrapper.generate(
                prompt=f"Original Prompt: {current_prompt}\n\nTask: {full_instruction}", 
                system_instruction="You are a creative Prompt Engineering Architect.", 
                temperature=1.0, 
                max_new_tokens=250
            )
            clean_prompt = new_prompt.replace('"', '').replace("Original Prompt:", "").strip()
            candidates.append(clean_prompt)
            
        return candidates

    # ---------------- Pipeline ----------------
    def optimize(self, initial_prompt, num_rounds=3):
        all_data = []
        yield {"type": "progress", "message": "Generating baseline..."}
        
        # Initial Setup
        best_output = self.model_wrapper.generate(initial_prompt, max_new_tokens=100)
        baseline_anchor = best_output
        
        # Current "Working" Best (used for evolution)
        current_best_prompt = initial_prompt
        current_best_score = 0.0
        current_best_output = best_output

        # GLOBAL Best (The absolute peak performance seen so far)
        global_best = {
            "prompt": initial_prompt,
            "score": 0.0,
            "output": best_output,
            "round": 0
        }

        for rnd in range(1, num_rounds + 1):
            yield {"type": "progress", "message": f"=== Round {rnd} ==="}
            
            # Diagnose based on the current working best
            current_stats = self.compute_final_score(current_best_prompt, current_best_output, baseline_anchor)
            root_cause = self.diagnose_root_cause(current_best_prompt, current_best_output, current_stats["penalties"])
            prescribed_strategy = self.strategy_map.get(root_cause, "CO_STAR_Framework")
            
            yield {"type": "progress", "message": f"Diagnosis: {root_cause} -> Strategy: {prescribed_strategy}"}

            # Generate Candidates
            candidates = self.apply_strategy(current_best_prompt, root_cause, prescribed_strategy)

            round_candidates = []
            for idx, cand_prompt in enumerate(candidates, 1):
                yield {"type": "progress", "message": f"Testing variation {idx}..."}
                
                cand_output = self.model_wrapper.generate(cand_prompt, max_new_tokens=120)
                score_data = self.compute_final_score(cand_prompt, cand_output, baseline_anchor)
                
                row = {
                    "Round": rnd, "Prompt No.": idx, "Strategy": prescribed_strategy,
                    "Prompt": cand_prompt, "Output": cand_output, 
                    "Final Score": score_data["final_score"],
                    "Penalties": score_data["penalties"]
                }
                all_data.append(row)
                round_candidates.append(row)

                # Check if this specific candidate is a new GLOBAL BEST
                if score_data["final_score"] > global_best["score"]:
                    global_best = {
                        "prompt": cand_prompt,
                        "score": score_data["final_score"],
                        "output": cand_output,
                        "round": rnd
                    }

            # Selection for NEXT round's parent
            # We sort to find the winner of THIS round
            round_candidates.sort(key=lambda x: x["Final Score"], reverse=True)
            round_winner = round_candidates[0]
            
            # Logic: We update the 'current_best' (parent for next round) 
            # only if it's decent. If it crashed hard, we might stick to previous,
            # but usually we want to keep evolving.
            current_best_prompt = round_winner["Prompt"]
            current_best_output = round_winner["Output"]
            
            if round_winner["Final Score"] >= current_best_score:
                current_best_score = round_winner["Final Score"]
                yield {"type": "progress", "message": f"Round {rnd} Winner Score: {current_best_score:.4f}"}
            else:
                yield {"type": "progress", "message": f"Score dropped this round. (Best: {global_best['score']:.4f})"}

        # === FINAL RESULT IS THE GLOBAL PEAK ===
        # Even if Round 5 crashed, we return the winner from Round X
        yield {
            "type": "result", 
            "best_prompt": global_best["prompt"], 
            "best_score": global_best["score"],
            "all_candidates": pd.DataFrame(all_data).fillna(0).to_dict(orient="records")
        }