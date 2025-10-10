import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class TextModel:
    def __init__(self, model_name="mosaicml/mpt-1b-redpajama-200b-dolly"):
        print(f"Loading model: {model_name}")

        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        offload_dir = "./offload"
        os.makedirs(offload_dir, exist_ok=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            offload_folder=offload_dir,
            trust_remote_code=True
        )

        print(f"Model loaded on device: {self.device} (float16 mode)")

    def generate(self, prompt, max_new_tokens=120):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

