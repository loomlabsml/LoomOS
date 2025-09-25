# HF Model Adapter

from transformers import pipeline

@loom_block("hf-model")
class HFModelAdapter:
    def __init__(self):
        self.pipe = pipeline("text-generation", model="distilgpt2")

    def predict(self, input_text: str) -> str:
        return self.pipe(input_text)[0]['generated_text']