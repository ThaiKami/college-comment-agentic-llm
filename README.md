# college-comment-agentic-llm

Runnable prototype of an agentic LLM pipeline for student comments. It uses a local vLLM server that exposes the OpenAI API.

## Setup

1. Start vLLM OpenAI-compatible server (example):

```bash
vllm serve your-model --host 0.0.0.0 --port 8000
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
export VLLM_MODEL=your-model
export VLLM_EMBEDDING_MODEL=your-embedding-model
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=local

python agentic_llm.py \
  --input data/comments.jsonl \
  --knowledge data/knowledge.docx \
  --output output.json
```

### Notes
- Input must be JSONL; each line can be a JSON object or plain string.
- Comment field defaults to `comment`, `text`, or `content`, or use `--field`.
- Requires a DOCX knowledge file; it is chunked and retrieved with RAG.
- Set embeddings model with `VLLM_EMBEDDING_MODEL` or `--embedding-model`.
- Control retrieval size with `--top-k`, and chunking with `--chunk-size`/`--chunk-overlap`.
- To append newly learned rules back to a doc file, pass `--update-kb path/to/file.txt`.

## Prompt Optimization (LLMsGreenRec-like)

This variant searches for a better system prompt using a multi-agent loop (evaluate, judge, detect error, infer reasons, refine, augment, select).

```bash
export VLLM_MODEL=your-model
export VLLM_EMBEDDING_MODEL=your-embedding-model
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=local

python comment_agentic/main.py \
  --train data/comments.jsonl \
  --knowledge data/knowledge.docx \
  --output prompt_search_output.json
```

Optional: pass a validation set with `--val data/valid.jsonl`.
