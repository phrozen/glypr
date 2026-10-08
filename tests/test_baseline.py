"""Unit tests for EmbeddingGemma 2 baseline embedder."""

import os
import pytest
import numpy as np
import torch

from baseline.embedder import EmbeddingGemmaBaseline

@pytest.fixture(scope="module")
def embedder():
    # Initialize text-only embedder for fast tests
    return EmbeddingGemmaBaseline(mode="text_only")

def test_unit_length_normalization(embedder):
    """Verify that all MRL truncation levels produce unit-norm vectors (within bfloat16 tolerance)."""
    text = "Unit length normalization test across Matryoshka dimensions."
    
    for dim in [768, 512, 256, 128]:
        emb = embedder.encode_query(text, task="SearchQuery", truncate_dim=dim)
        assert emb.shape[-1] == dim
        norm = np.linalg.norm(emb)
        # bfloat16 has ~7-8 bits of mantissa (~7.8e-3 precision)
        assert np.isclose(norm, 1.0, atol=5e-3), f"Norm at dim={dim} is {norm}, expected 1.0"


def test_asymmetric_retrieval(embedder):
    """Verify asymmetric search query vs titled document matching."""
    query = "How to write unit tests in Python?"
    match_doc = "pytest is a popular framework for writing unit tests in Python."
    distractor_doc = "Baking sourdough bread requires flour, water, and active starter."

    q_emb = embedder.encode_query(query, task="QuestionAnswering")
    doc_match_emb = embedder.encode_document(match_doc, title="Testing Guide")
    doc_distractor_emb = embedder.encode_document(distractor_doc, title="Baking 101")

    sim_match = embedder.similarity(q_emb, doc_match_emb).item()
    sim_distractor = embedder.similarity(q_emb, doc_distractor_emb).item()

    assert sim_match > sim_distractor + 0.25, f"Expected match ({sim_match}) >> distractor ({sim_distractor})"


def test_mrl_monotonic_correlation(embedder):
    """Verify 512d and 256d retain high similarity correlation with 768d."""
    query = "Distributed consensus algorithm Paxos"
    doc = "Paxos ensures agreement among a network of unreliable processors."

    sims = {}
    for dim in [768, 512, 256, 128]:
        q = embedder.encode_query(query, task="SearchQuery", truncate_dim=dim)
        d = embedder.encode_document(doc, title="Distributed Systems", truncate_dim=dim)
        sims[dim] = embedder.similarity(q, d).item()

    # 512d should be virtually identical to 768d (delta < 0.02)
    assert abs(sims[768] - sims[512]) < 0.02, f"512d delta too large: {sims[768]} vs {sims[512]}"
    # 256d should stay close (delta < 0.05)
    assert abs(sims[768] - sims[256]) < 0.05, f"256d delta too large: {sims[768]} vs {sims[256]}"

def test_strict_float16_rejection():
    """Verify that float16 is strictly rejected to prevent NaN issues."""
    with pytest.raises(ValueError, match="float16 is strictly forbidden"):
        EmbeddingGemmaBaseline(mode="text_only", torch_dtype=torch.float16)
