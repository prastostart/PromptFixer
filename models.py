# models.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class TextModel:
    def __init__(self, model_name="microsoft/phi-2"):
        print(f"Loading model: {model_name}")
        # Set device: MPS if available, otherwise CPU
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model in full precision for stability
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        print(f"Model loaded on device: {self.device} (float32 for stability)")

    def generate(self, prompt, max_length=100):
        # Tokenize input and send tensors to the correct device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate output safely
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

