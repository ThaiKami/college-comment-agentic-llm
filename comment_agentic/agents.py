import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import OpenAI

from opt.utils import coerce_score, extract_wrapped_text, parse_json


@dataclass
class LLMConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float
    request_timeout: int


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def chat(self, system: str, user: str, temperature: float = None) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=temperature if temperature is not None else self.config.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            timeout=self.config.request_timeout,
        )
        return response.choices[0].message.content or ""


class EvalAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(self, prompt: str, context: str, comment: str) -> Dict[str, Any]:
        system = prompt.strip()
        user = (
            "Retrieved Context:\n"
            f"{context}\n\n"
            "Task: Produce an actionable report for the student comment.\n"
            "Return JSON with keys: topic, sentiment, confidence, rationale, actions.\n"
            "- sentiment: positive|neutral|negative|mixed\n"
            "- confidence: 0 to 1\n"
            "- actions: list of short action strings\n\n"
            "Student Comment:\n"
            f"{comment}"
        )
        return parse_json(self.llm.chat(system, user))


class ReportJudgeAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def score(self, context: str, comment: str, report: Dict[str, Any]) -> Dict[str, Any]:
        system = "You are a strict evaluator. Output JSON only."
        user = (
            "Retrieved Context:\n"
            f"{context}\n\n"
            "Task: Score the report for correctness, usefulness, and alignment with the comment.\n"
            "Return JSON with keys: score (0 to 1), issues (list of strings).\n"
            "Student Comment:\n"
            f"{comment}\n\n"
            "Report JSON:\n"
            f"{json.dumps(report, ensure_ascii=True)}"
        )
        return parse_json(self.llm.chat(system, user))


class DetectErrorAgent:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def detect(self, judge: Dict[str, Any]) -> bool:
        score = coerce_score(judge.get("score"), default=0.0)
        issues = judge.get("issues", [])
        return score < self.threshold or (isinstance(issues, list) and len(issues) > 0)


class InferReasonAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def infer(self, prompt: str, comment: str, judge: Dict[str, Any], max_reasons: int) -> List[str]:
        system = "You analyze root causes. Output JSON only."
        user = (
            "Task: Provide root causes for why the report failed.\n"
            "Return JSON with key: reasons (list of strings).\n"
            f"Max reasons: {max_reasons}\n\n"
            "Prompt:\n"
            f"{prompt}\n\n"
            "Student Comment:\n"
            f"{comment}\n\n"
            "Judge JSON:\n"
            f"{json.dumps(judge, ensure_ascii=True)}"
        )
        data = parse_json(self.llm.chat(system, user))
        reasons = data.get("reasons", [])
        return [r for r in reasons if isinstance(r, str)]


class RefinePromptAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def refine(self, prompt: str, comment: str, reasons: List[str]) -> str:
        system = "You improve prompts."
        user = (
            "Current prompt:\n"
            f"{prompt}\n\n"
            "Student Comment:\n"
            f"{comment}\n\n"
            "Root cause reasons:\n"
            f"{json.dumps(reasons, ensure_ascii=True)}\n\n"
            "Task: Write an improved system prompt to produce better actionable reports.\n"
            "Wrap the improved prompt with <START> and <END>."
        )
        response = self.llm.chat(system, user)
        wrapped = extract_wrapped_text(response)
        return wrapped or response.strip()


class AugmentAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def augment(self, prompt: str, sample_size: int) -> List[str]:
        system = "You paraphrase prompts."
        user = (
            "Task: Generate a variation of this prompt while preserving meaning.\n"
            f"Prompt:\n{prompt}\n\n"
            "Output only the rewritten prompt."
        )
        return [self.llm.chat(system, user).strip() for _ in range(sample_size)]


class SelectionAgent:
    def select(
        self,
        prompts: List[str],
        train_data: List[Dict[str, Any]],
        evaluator,
        judge,
        context_fetcher,
        time_steps: int,
        explore_param: float,
        sample_num: int,
        beam_width: int,
    ) -> List[str]:
        if not prompts:
            return []
        selections = [0] * len(prompts)
        rewards = [0.0] * len(prompts)

        for t in range(1, time_steps + 1):
            sample_data = random.sample(train_data, min(sample_num, len(train_data)))
            ucb_values = []
            for i in range(len(prompts)):
                if selections[i] == 0:
                    ucb_values.append(float("inf"))
                else:
                    ucb_values.append(
                        rewards[i] / selections[i]
                        + explore_param * ((t ** 0.5) / selections[i])
                    )
            selected_idx = ucb_values.index(max(ucb_values))
            prompt = prompts[selected_idx]

            for item in sample_data:
                comment = item["comment"]
                context = context_fetcher(comment)
                report = evaluator.evaluate(prompt, context, comment)
                judge_out = judge.score(context, comment, report)
                score = coerce_score(judge_out.get("score"), default=0.0)
                rewards[selected_idx] += score
                selections[selected_idx] += 1

        prompt_reward_pairs = list(zip(rewards, prompts))
        prompt_reward_pairs.sort(reverse=True, key=lambda x: x[0])
        return [pair[1] for pair in prompt_reward_pairs[:beam_width]]
