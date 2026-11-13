"""EmbeddingGemma 2: one shared space for spoken audio, queries and documents.

Every call runs on one dedicated worker thread (MPS is not happy with concurrent use), so callers in asyncio use
`await embedder.aencode(...)`.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config

_model = None
_qmodel = None
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")      # MPS: docs + audio
_qpool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed-q")   # CPU text-only copy: queries (critical path)


def model():
    global _model
    with _lock:
        if _model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dtype = torch.bfloat16 if config.EMBED_DEVICE in ("mps", "cuda") else torch.float32
            _model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE,
                                         model_kwargs={"torch_dtype": dtype})
        return _model


def qmodel():
    """Text-only CPU copy (271M) for query embeddings, so a query never waits behind document/audio batches on MPS."""
    global _qmodel
    with _lock:
        if _qmodel is None:
            import torch
            from sentence_transformers import SentenceTransformer
            _qmodel = SentenceTransformer(config.EMBED_MODEL, device="cpu", model_kwargs={"torch_dtype": torch.float32},
                                          config_kwargs={"vision_config": None, "audio_config": None})
        return _qmodel


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if config.EMBED_DIM < x.shape[-1]:
        x = x[..., : config.EMBED_DIM]
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def encode_queries(texts: list[str]) -> np.ndarray:
    return _norm(qmodel().encode(texts, prompt_name="SearchQuery", batch_size=16))


def encode_docs(texts: list[str], titles: list[str] | None = None) -> np.ndarray:
    if titles:
        texts = [f"title: {t or 'none'} | text: {x}" for t, x in zip(titles, texts)]
        return _norm(model().encode(texts, batch_size=16))
    return _norm(model().encode(texts, prompt_name="Document", batch_size=16))


def encode_audio(wavs: list[np.ndarray]) -> np.ndarray:
    """Mono float32 16 kHz arrays -> embeddings (no prompt for audio)."""
    return _norm(model().encode([{"audio": w.astype(np.float32)} for w in wavs], batch_size=4))


def warmup():
    t = time.perf_counter()
    model()
    qmodel()
    for n in (1, 2, 3, 5, 8):
        encode_queries(["warm up query"] * n)
        encode_docs(["warm up document text"] * n)
    rng = np.random.default_rng(0)
    for sec in (1.0, 2.0, 3.0):
        encode_audio([rng.standard_normal(int(16000 * sec)).astype(np.float32) * 0.01])
    return time.perf_counter() - t


async def arun(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_qpool if fn is encode_queries else _pool, fn, *args)
