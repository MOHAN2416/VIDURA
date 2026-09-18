"""VIDURA Local Embedding Providers (Phase 12).

Provides local dense vector embeddings for semantic retrieval.
Supports Ollama local embeddings when available, with an offline deterministic
dense vector generator ensuring 100% offline capability and test repeatability.
"""
from __future__ import annotations

import abc
import hashlib
import logging
import math
import re
from typing import Any

from rag.models import EmbeddingVector
from config import load_config

logger = logging.getLogger("VIDURA.rag.embeddings")


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculates cosine similarity between two numeric vectors, bounded in [0.0, 1.0]."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, (sim + 1.0) / 2.0 if sim < 0.0 else sim))


class BaseEmbeddingProvider(abc.ABC):
    """Abstract base class for all embedding providers in VIDURA."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Returns the embedding vector dimension."""
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Returns the model name identifier."""
        pass

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Returns the provider name identifier."""
        pass

    @abc.abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generates a dense embedding vector for the provided text."""
        pass

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a batch of texts."""
        return [self.embed_text(t) for t in texts]

    def embed(self, text: str) -> EmbeddingVector | None:
        """Generates an EmbeddingVector instance for the provided text, or None on failure."""
        try:
            vec = self.embed_text(text)
            return EmbeddingVector(
                vector=vec,
                dimension=self.dimension,
                model=self.model_name,
                provider=self.provider_name,
            )
        except Exception:
            return None


class DeterministicLocalEmbeddingProvider(BaseEmbeddingProvider):
    """Offline, deterministic dense vector generator for codebase semantic retrieval.
    
    Uses subword character n-grams and token hashing with L2 unit normalization.
    Guarantees reproducible, continuous semantic similarity with zero network calls.
    """

    def __init__(self, dimension: int = 128, model_name: str = "vidura-local-hash-128") -> None:
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return "deterministic_local"

    def embed_text(self, text: str) -> list[float]:
        """Maps text into a normalized dense vector in R^dimension."""
        if not text or not text.strip():
            return [0.0] * self._dimension

        vec = [0.0] * self._dimension
        raw_tokens = re.findall(r"[a-zA-Z0-9_\-\.]{2,}", text.lower())

        if not raw_tokens:
            return [0.0] * self._dimension

        # Collect compound identifiers as well as individual sub-words (snake_case, dot, dash)
        all_tokens: list[str] = []
        for tok in raw_tokens:
            all_tokens.append(tok)
            sub_words = re.findall(r"[a-zA-Z0-9]{2,}", tok)
            if len(sub_words) > 1:
                all_tokens.extend(sub_words)

        # 1. Word token hashing
        for token in all_tokens:
            weight = 1.0
            if len(token) > 5:
                weight = 1.5
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dimension
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[idx] += sign * weight

            # 2. Subword character n-grams (3-grams and 4-grams)
            if len(token) >= 4:
                for i in range(len(token) - 2):
                    ngram = token[i : i + 3]
                    nh = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16)
                    nidx = nh % self._dimension
                    nsign = 1.0 if ((nh >> 6) & 1) == 0 else -1.0
                    vec[nidx] += nsign * 0.4

        # 3. L2 Normalization
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]

        return vec


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Generates embeddings using a local Ollama service."""

    def __init__(
        self,
        host: str | None = None,
        model_name: str = "all-minilm",
        dimension: int = 384,
    ) -> None:
        cfg = load_config()
        self._host = host or cfg.ollama_host
        self._model_name = model_name or cfg.vidura_rag_embedding_model
        self._dimension = dimension
        self._client: Any = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return "ollama_local"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self._host)
            except Exception as err:
                logger.warning(f"Failed to initialize ollama.Client: {err}")
                self._client = None
        return self._client

    def embed_text(self, text: str) -> list[float]:
        """Calls local Ollama embeddings API."""
        client = self._get_client()
        if client is None:
            raise RuntimeError("Ollama client unavailable")

        try:
            res = client.embeddings(model=self._model_name, prompt=text)
            embedding = res.get("embedding") if isinstance(res, dict) else getattr(res, "embedding", None)
            if embedding and isinstance(embedding, list):
                self._dimension = len(embedding)
                return [float(x) for x in embedding]
        except Exception as err:
            logger.warning(f"Ollama embedding request failed: {err}")
            raise RuntimeError(f"Ollama embedding error: {err}")

        raise RuntimeError("No embedding returned by Ollama")


class EmbeddingManager:
    """Manages active embedding provider selection, fallback handling, and vector caching."""

    def __init__(
        self,
        provider: BaseEmbeddingProvider | None = None,
        model_name: str | None = None,
        dimension: int = 128,
        enable_ollama_fallback: bool = True,
    ) -> None:
        self.fallback_provider = DeterministicLocalEmbeddingProvider(
            dimension=dimension,
            model_name=f"deterministic-local-{dimension}d",
        )
        self.enable_fallback = enable_ollama_fallback
        self._active_provider: BaseEmbeddingProvider

        if provider:
            self._active_provider = provider
        elif model_name:
            if "ollama" in model_name or "minilm" in model_name or "nomic" in model_name:
                self._active_provider = OllamaEmbeddingProvider(model_name=model_name, dimension=dimension)
            else:
                self._active_provider = DeterministicLocalEmbeddingProvider(
                    dimension=dimension,
                    model_name=f"deterministic-local-{dimension}d",
                )
        else:
            cfg = load_config()
            cfg_model = getattr(cfg, "vidura_rag_embedding_model", "all-minilm")
            self._active_provider = DeterministicLocalEmbeddingProvider(
                dimension=dimension,
                model_name=f"deterministic-local-{dimension}d",
            )

        self._cache: dict[str, list[float]] = {}

    @property
    def active_provider(self) -> BaseEmbeddingProvider:
        return self._active_provider

    def set_provider(self, provider: BaseEmbeddingProvider) -> None:
        self._active_provider = provider
        self._cache.clear()

    def embed_text(self, text: str) -> list[float]:
        """Generates embedding for text using active provider with automatic offline fallback."""
        if not text:
            return [0.0] * self._active_provider.dimension

        cache_key = f"{self._active_provider.model_name}:{hashlib.md5(text.encode('utf-8')).hexdigest()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            vec = self._active_provider.embed_text(text)
        except Exception as err:
            if self.enable_fallback and self._active_provider != self.fallback_provider:
                logger.info(f"Primary embedding failed ({err}); using deterministic local fallback.")
                vec = self.fallback_provider.embed_text(text)
            else:
                raise

        # Cache vector
        if len(self._cache) > 2000:
            self._cache.clear()
        self._cache[cache_key] = vec
        return vec

    def embed(self, text: str) -> EmbeddingVector:
        """Generates an EmbeddingVector using active provider or fallback."""
        vec = self.embed_text(text)
        return EmbeddingVector(
            vector=vec,
            dimension=len(vec),
            model=self._active_provider.model_name,
            provider=self._active_provider.provider_name,
        )

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Helper to directly compute cosine similarity between two text snippets."""
        vec_a = self.embed_text(text_a)
        vec_b = self.embed_text(text_b)
        return cosine_similarity(vec_a, vec_b)
