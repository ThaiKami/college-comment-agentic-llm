import argparse
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from openai import OpenAI

from agents import (
    AugmentAgent,
    DetectErrorAgent,
    EvalAgent,
    InferReasonAgent,
    LLMClient,
    LLMConfig,
    RefinePromptAgent,
    ReportJudgeAgent,
    SelectionAgent,
)
from rag import RagConfig, RagIndex, format_context, read_docx


def load_knowledge(path: str) -> str:
    return read_docx(path)


def load_jsonl(path: str, field: Optional[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                data = {"comment": line}

            if isinstance(data, dict):
                comment = None
                if field and field in data:
                    comment = data[field]
                else:
                    for key in ("comment", "text", "content"):
                        if key in data:
                            comment = data[key]
                            break
                if comment is None:
                    comment = json.dumps(data, ensure_ascii=True)
            else:
                comment = str(data)

            items.append(
                {"id": data.get("id") if isinstance(data, dict) else None, "comment": str(comment)}
            )
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic prompt optimization for student comments"
    )
    parser.add_argument(
        "--provider",
        default="vllm",
        choices=["vllm", "openai"],
        help="LLM provider",
    )
    parser.add_argument("--train", required=True, help="Path to JSONL training comments")
    parser.add_argument("--val", default=None, help="Optional JSONL validation comments")
    parser.add_argument("--knowledge", required=True, help="Path to knowledge/rules docx file")
    parser.add_argument(
        "--output", default="prompt_search_output.json", help="Path to output JSON"
    )
    parser.add_argument("--field", default=None, help="JSON field containing the comment text")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--embedding-model", default=None, help="Embedding model")
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--embedding-base-url", default=None, help="Embedding API base URL"
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--search-depth", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-feedbacks", type=int, default=2)
    parser.add_argument("--error-size", type=int, default=6)
    parser.add_argument("--augmentation", type=int, default=2)
    parser.add_argument("--time-steps", type=int, default=12)
    parser.add_argument("--explore-param", type=float, default=2.0)
    parser.add_argument("--sample-num", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


DEFAULT_PROMPT = (
    "You are an analyst of student feedback."
    " Always produce JSON only with keys: topic, sentiment, confidence, rationale, actions."
    " Focus on actionable, specific recommendations for instructors."
)


def compute_average_score(
    prompt: str,
    data: List[Dict[str, Any]],
    context_fetcher,
    evaluator: EvalAgent,
    judge: ReportJudgeAgent,
) -> float:
    scores: List[float] = []
    for item in data:
        context = context_fetcher(item["comment"])
        report = evaluator.evaluate(prompt, context, item["comment"])
        judged = judge.score(context, item["comment"], report)
        try:
            scores.append(float(judged.get("score", 0.0)))
        except (TypeError, ValueError):
            scores.append(0.0)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)
    if args.provider == "openai":
        model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        embedding_model = args.embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        base_url = args.base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        embedding_base_url = args.embedding_base_url or os.getenv(
            "OPENAI_EMBEDDING_BASE_URL", base_url
        )
        api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise SystemExit("Missing OpenAI API key. Set --api-key or OPENAI_API_KEY.")
    else:
        model = args.model or os.getenv("VLLM_MODEL", "Qwen/Qwen3-1.7B")
        embedding_model = args.embedding_model or os.getenv(
            "VLLM_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
        )
        base_url = args.base_url or os.getenv(
            "VLLM_BASE_URL", "http://localhost:8000/v1"
        )
        embedding_base_url = args.embedding_base_url or os.getenv(
            "VLLM_EMBEDDING_BASE_URL", "http://localhost:8001/v1"
        )
        api_key = args.api_key or os.getenv("VLLM_API_KEY", "local")

    random.seed(args.seed)

    knowledge = load_knowledge(args.knowledge)
    train_data = load_jsonl(args.train, args.field)
    val_data = load_jsonl(args.val, args.field) if args.val else train_data

    config = LLMConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=args.temperature,
        request_timeout=args.timeout,
    )
    llm = LLMClient(config)

    embedding_client = OpenAI(base_url=embedding_base_url, api_key=api_key)
    rag = RagIndex(
        embedding_client,
        embedding_model,
        RagConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
        ),
    )
    rag.build(knowledge)
    logger.info("Built RAG index with %d chunks", len(rag.chunks))

    eval_agent = EvalAgent(llm)
    judge_agent = ReportJudgeAgent(llm)
    detect_agent = DetectErrorAgent(threshold=args.threshold)
    infer_agent = InferReasonAgent(llm)
    refine_agent = RefinePromptAgent(llm)
    augment_agent = AugmentAgent(llm)
    selection_agent = SelectionAgent()

    beam = [DEFAULT_PROMPT]
    prompt_history: List[str] = []

    def get_context(comment: str) -> str:
        return format_context(rag.retrieve(comment, args.top_k))

    def evaluate_case(prompt: str, item: Dict[str, Any]) -> Dict[str, Any]:
        context = get_context(item["comment"])
        report = eval_agent.evaluate(prompt, context, item["comment"])
        judge_out = judge_agent.score(context, item["comment"], report)
        return {"report": report, "judge": judge_out}

    for _ in range(args.search_depth):
        candidate_prompts: List[str] = []
        for prompt in beam:
            logger.info("Evaluating prompt in beam (%d items)", len(beam))
            batch = random.sample(train_data, min(args.batch_size, len(train_data)))
            error_cases = []

            with ThreadPoolExecutor(max_workers=min(16, len(batch) or 1)) as executor:
                results = list(executor.map(lambda item: evaluate_case(prompt, item), batch))

            for item, result in zip(batch, results):
                if detect_agent.detect(result["judge"]):
                    error_cases.append({"comment": item["comment"], "judge": result["judge"]})

            if not error_cases:
                logger.info("No error cases in this batch")
                continue

            errors_group = random.sample(error_cases, min(args.error_size, len(error_cases)))
            for err in errors_group:
                reasons = infer_agent.infer(prompt, err["comment"], err["judge"], args.num_feedbacks)
                refined = refine_agent.refine(prompt, err["comment"], reasons)
                augmented = augment_agent.augment(refined, args.augmentation)
                candidate_prompts.extend([refined] + augmented)

        prompt_history.extend(candidate_prompts)
        if candidate_prompts:
            logger.info("Selecting best prompts from %d candidates", len(candidate_prompts))
            beam = selection_agent.select(
                candidate_prompts,
                train_data,
                eval_agent,
                judge_agent,
                get_context,
                args.time_steps,
                args.explore_param,
                args.sample_num,
                args.beam_width,
            )

    scored = [
        {
            "prompt": prompt,
            "avg_score": compute_average_score(
                prompt, val_data, get_context, eval_agent, judge_agent
            ),
        }
        for prompt in beam
    ]
    scored.sort(key=lambda x: x["avg_score"], reverse=True)
    best_prompt = scored[0]["prompt"] if scored else DEFAULT_PROMPT
    logger.info("Best prompt average score: %.3f", scored[0]["avg_score"] if scored else 0.0)

    output = {
        "best_prompt": best_prompt,
        "beam": scored,
        "history": prompt_history,
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
