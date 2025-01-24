import os
import openai
from typing import Dict


class OpenAIWrapper:
    def __init__(self, api_key: str = None, engine: str = "gpt-4o-mini",
                 temperature: float = 0.0, max_tokens: int = 100):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in the environment variable OPENAI_API_KEY.")

        openai.api_key = self.api_key
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_prompt(self, example: Dict, template: str) -> str:
        try:
            return template.format(**example)
        except KeyError as e:
            raise ValueError(f"Missing key {e} in example dictionary.")

    def generate_response(self, prompt: str) -> str:
        try:
            response = openai.Completion.create(
                engine=self.engine,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
