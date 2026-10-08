"""Baseline embedding harness for google/embeddinggemma-2 using SentenceTransformers."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"

def get_optimal_dtype() -> torch.dtype:
    """Determine safe numerical precision. Strictly blocks float16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon MPS supports bfloat16 in recent PyTorch versions
        return torch.bfloat16
    else:
        return torch.float32

class EmbeddingGemmaBaseline:
    """Ground-truth Python reference wrapper for EmbeddingGemma 2."""

    def __init__(
        self,
        mode: str = "text_only",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """Initialize EmbeddingGemma 2 in a specified modular configuration.

        Args:
            mode: One of 'text_only' (270M), 'text_vision' (440M), 'full' (740M).
            device: 'mps', 'cuda', 'cpu', or None (auto-detected).
            torch_dtype: torch.bfloat16 or torch.float32. float16 is strictly rejected.
        """
        self.mode = mode
        self.dtype = torch_dtype or get_optimal_dtype()

        if self.dtype == torch.float16:
            raise ValueError(
                "CRITICAL: float16 is strictly forbidden for EmbeddingGemma 2 due to activation dynamic range overflow/underflow. Use torch.bfloat16 or torch.float32."
            )

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        config_kwargs: Dict[str, Any] = {}
        if mode == "text_only":
            config_kwargs = {"vision_config": None, "audio_config": None}
        elif mode == "text_vision":
            config_kwargs = {"audio_config": None}
        elif mode == "full":
            config_kwargs = {}
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'text_only', 'text_vision', or 'full'.")

        print(f"Loading EmbeddingGemma 2 ({mode}) on device={self.device}, dtype={self.dtype}...")
        self.model = SentenceTransformer(
            MODEL_ID,
            model_kwargs={"torch_dtype": self.dtype},
            config_kwargs=config_kwargs,
            device=self.device,
        )

    def encode_query(
        self,
        query: Union[str, List[str]],
        task: str = "SearchQuery",
        truncate_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Encode a retrieval query with task instruction prefix.

        Supported tasks: 'SearchQuery', 'CodeRetrieval', 'QuestionAnswering', 'FactChecking'.
        """
        return self.model.encode(
            query,
            prompt_name=task,
            truncate_dim=truncate_dim,
            normalize_embeddings=True,
        )

    def encode_document(
        self,
        content: Union[str, List[str]],
        title: Optional[Union[str, List[str]]] = None,
        truncate_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Encode document passages with proper title syntax."""
        if title is None:
            # Use 'Document' prompt_name which prepends 'title: none | text: '
            return self.model.encode(
                content,
                prompt_name="Document",
                truncate_dim=truncate_dim,
                normalize_embeddings=True,
            )
        else:
            if isinstance(content, str) and isinstance(title, str):
                formatted = f"title: {title} | text: {content}"
            elif isinstance(content, list) and isinstance(title, list):
                formatted = [f"title: {t} | text: {c}" for t, c in zip(title, content)]
            else:
                raise ValueError("content and title must both be str or list")

            return self.model.encode(
                formatted,
                truncate_dim=truncate_dim,
                normalize_embeddings=True,
            )

    def encode_symmetric(
        self,
        texts: Union[str, List[str]],
        task: str = "SentenceSimilarity",
        truncate_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Encode inputs for symmetric tasks: 'SentenceSimilarity', 'Classification', 'Clustering'."""
        return self.model.encode(
            texts,
            prompt_name=task,
            truncate_dim=truncate_dim,
            normalize_embeddings=True,
        )

    def encode_media(
        self,
        media_input: Union[Dict[str, Any], List[Dict[str, Any]]],
        truncate_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Encode image, audio, video, or interleaved dictionary inputs."""
        return self.model.encode(
            media_input,
            truncate_dim=truncate_dim,
            normalize_embeddings=True,
        )

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between two sets of embeddings."""
        return self.model.similarity(emb1, emb2).cpu().numpy()
