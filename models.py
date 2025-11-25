# models.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class TextModel:
    def __init__(self, model_name="mosaicml/mpt-1b-instruct"):
        print(f"Loading model: {model_name}")

        # Use MPS if available, else CPU
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Optional: offload directory for large models
        offload_dir = "./offload"
        os.makedirs(offload_dir, exist_ok=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",                # automatically use MPS/CPU
            torch_dtype=torch.float16,        # reduce memory usage
            low_cpu_mem_usage=True,           # helps on Mac
            trust_remote_code=True
        )

        print(f"Model loaded on device: {self.device} (float16 mode)")

    def generate(self, prompt, max_new_tokens=120, temperature=0.3):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
