from abc import ABC, abstractmethod
import logging

from anthropic import Anthropic
from openai import OpenAI
import transformers
import torch


OPENAI_KEY = "YOUR KEY"
ANTHROPIC_KEY = "YOUR KEY"
HUGGINGFACE_KEY = "YOUR KEY"

LLM_FACTORY = {
    "gpt-3.5-turbo": lambda: ChatGPT("gpt-3.5-turbo"),
    "gpt-4-turbo": lambda: ChatGPT("gpt-4-turbo"),
    "gpt-4": lambda: ChatGPT("gpt-4"),
    "gpt-4o": lambda: ChatGPT("gpt-4o"),
    "gpt-5": lambda: ChatGPT("gpt-5"),
    "claude-3.5": lambda: Claude("claude-3-5-sonnet-20240620"),
    "llama-3-8b": lambda: HuggingFace("meta-llama/Meta-Llama-3-8B-Instruct"),
    "gemma-2-9b": lambda: HuggingFace("google/gemma-2-9b-it"),
}


class LLM(ABC):
    def __init__(self, name, max_len=1024):
        self.name = name
        self.max_len = max_len

    @abstractmethod
    def query(self, question: str) -> str:
        raise NotImplementedError


import os

class ChatGPT(LLM):
    PRICES = {
        "gpt-3.5-turbo": (0.5/1e6, 1.5/1e6),
        "gpt-4":         (30/1e6, 60/1e6),
        "gpt-4-turbo":   (10/1e6, 30/1e6),
        "gpt-4o":        (5/1e6, 15/1e6),
        "gpt-5":         (1.25/1e6, 10/1e6),
    }
    TOTAL_COST = 0

    def __init__(self, name, max_len=1024):
        assert name in {"gpt-3.5-turbo","gpt-4","gpt-4-turbo","gpt-4o","gpt-5"}
        super().__init__(name, max_len=max_len)
        self.client = OpenAI(api_key=OPENAI_KEY)

    def _is_reasoning_model(self):
        return self.name.startswith(("gpt-5","o"))

    def _extract_tokens(self, usage):
        inp = getattr(usage, "prompt_tokens", None)
        if inp is None: inp = getattr(usage, "input_tokens", 0)
        out = getattr(usage, "completion_tokens", None)
        if out is None: out = getattr(usage, "output_tokens", 0)
        return int(inp or 0), int(out or 0)

    def query(self, question: str) -> str:
        if self._is_reasoning_model():  # gpt-5 / o* 走 Responses API
            response = self.client.responses.create(
                model=self.name,
                input=question,
                max_output_tokens=self.max_len,
                reasoning = {"effort": "minimal"}
            )
            answer = getattr(response, "output_text", "") or ""
            if not answer and getattr(response, "output", None):
                parts = []
                for item in response.output:
                    content = getattr(item, "content", None)
                    if not content:
                        continue
                    for block in content:
                        if getattr(block, "type", "") in ("output_text", "text"):
                            text = getattr(block, "text", "") or ""
                            if text:
                                parts.append(text)
                answer = "".join(parts)

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        else:  # 其余模型保持原样：Chat Completions
            response = self.client.chat.completions.create(
                model=self.name,
                temperature=0,
                top_p=0,
                max_tokens=self.max_len,
                messages=[{"role": "user", "content": question}],
            )
            answer = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        cost = ChatGPT.PRICES[self.name][0] * input_tokens + ChatGPT.PRICES[self.name][1] * output_tokens
        self.TOTAL_COST += cost
        logging.info("===== USAGE =====")
        logging.info(f"input tokens: {input_tokens}; output tokens: {output_tokens}")
        logging.info(f"query cost: ${round(cost, 4)}; total cost: ${round(self.TOTAL_COST, 4)}")
        logging.info("===== USAGE =====")
        return answer


class Claude(LLM):
    def __init__(self, name:str, max_len=1024):
        assert name.startswith("claude"), "unsupported Claude version"
        super().__init__(name, max_len=max_len)
        self.client = Anthropic(api_key=ANTHROPIC_KEY)

    def query(self, question: str) -> str:
        message = self.client.messages.create(
            model=self.name,
            max_tokens=self.max_len,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }
            ]
        )
        return message.content



class HuggingFace(LLM):
    def __init__(self, name, max_len=1024):
        super().__init__(name, max_len=max_len)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            token=HUGGINGFACE_KEY
        )


    def query(self, question: str) -> str:
        messages = [
            {"role": "user", "content": question},
        ]

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_len,
            eos_token_id=terminators,
        )
        return outputs[0]["generated_text"][-1]["content"]
