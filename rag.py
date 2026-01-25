import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import faiss
from docx import Document
from openai import OpenAI


@dataclass
class RagConfig:
    chunk_size: int
    chunk_overlap: int
    top_k: int


class RagIndex:
    def __init__(self, client: OpenAI, model: str, config: RagConfig):
        self.client = client
        self.model = model
        self.config = config
        self.chunks: List[str] = []
        self.index = None

    def build(self, text: str) -> None:
        self.chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
        if not self.chunks:
            self.index = None
            return
        embeddings = embed_texts(self.client, self.model, self.chunks)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        self.index = index

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[int, str]]:
        if not self.index or not self.chunks:
            return []
        k = top_k or self.config.top_k
        embedding = embed_texts(self.client, self.model, [query])
        distances, indices = self.index.search(embedding, min(k, len(self.chunks)))
        results = []
        for idx in indices[0]:
            if idx < 0:
                continue
            results.append((int(idx), self.chunks[int(idx)]))
        return results


def read_docx(path: str) -> str:
    doc = Document(path)
    parts = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0:
        return [text]
    overlap = max(0, min(chunk_overlap, chunk_size - 1))
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def embed_texts(client: OpenAI, model: str, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    response = client.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def format_context(chunks: List[Tuple[int, str]]) -> str:
    if not chunks:
        return "(no relevant context)"
    formatted = [f"[chunk {idx}] {text}" for idx, text in chunks]
    return "\n".join(formatted)
