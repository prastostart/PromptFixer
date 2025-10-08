# models.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class TextModel:
    def __init__(self, model_name="mosaicml/mpt-1b-redpajama-200b-dolly"):
        print(f"Loading model: {model_name}")

        # Device: MPS if available, else CPU
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Ensure offload folder exists
        offload_dir = "./offload"
        os.makedirs(offload_dir, exist_ok=True)

        # Load model safely with device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,      # float16 for reduced memory
            device_map="auto",        # automatically map layers to available devices
            offload_folder=offload_dir,
            trust_remote_code=True
        )

        print(f"Model loaded on device: {self.device} (float16 for stability)")

    def generate(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        max_new_tokens = 150  # ✅ define default generation length
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

