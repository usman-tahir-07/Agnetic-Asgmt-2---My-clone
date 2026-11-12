"""
Hybrid RAG core: chunking, BM25 retrieval, dense retrieval and Reciprocal Rank Fusion.

Pipeline:  load docs -> chunk (~250 words, overlap) -> index twice (BM25 + embeddings)
           -> retrieve (bm25 | dense | hybrid/RRF) -> top-k chunks with scores.

Embeddings are precomputed once by build_index.py and stored in embeddings.npy
(plus chunks.json with the chunk text + metadata), so the Space only loads them.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHUNKS_PATH = ROOT / "chunks.json"
EMB_PATH = ROOT / "embeddings.npy"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking parameters (in words)
TARGET_WORDS = 250   # aim for ~200-300 words per chunk
MAX_WORDS = 300
OVERLAP_WORDS = 40   # small overlap carried from the previous chunk

RRF_K = 60           # constant from the RRF paper (Cormack et al., 2009)


# ----------------------------------------------------------------------------
# 1. Load and chunk
# ----------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str      # e.g. "faq.md#3"
    source: str        # file name
    index: int         # position of the chunk inside its file
    text: str


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # only needed if you add PDFs
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines; break any paragraph longer than MAX_WORDS into pieces."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    for p in paras:
        words = p.split()
        if len(words) <= MAX_WORDS:
            out.append(p)
        else:
            step = TARGET_WORDS
            for i in range(0, len(words), step):
                out.append(" ".join(words[i:i + step]))
    return out


def chunk_document(source: str, text: str) -> list[Chunk]:
    """Greedy paragraph packing into ~TARGET_WORDS chunks with OVERLAP_WORDS overlap.

    Paragraph-aware packing keeps FAQ question/answer pairs and sections together
    instead of cutting them in the middle of a sentence.
    """
    chunks: list[Chunk] = []
    current: list[str] = []          # words of the chunk being built
    fresh_words = 0                  # words added since the overlap was carried over

    def flush():
        nonlocal current, fresh_words
        if fresh_words == 0:
            return
        chunks.append(Chunk(f"{source}#{len(chunks)}", source, len(chunks), " ".join(current)))
        current = current[-OVERLAP_WORDS:]   # carry a small overlap into the next chunk
        fresh_words = 0

    for para in _paragraphs(text):
        words = para.split()
        if current and len(current) + len(words) > MAX_WORDS and fresh_words > 0:
            flush()
        current.extend(words)
        fresh_words += len(words)
        if len(current) >= TARGET_WORDS:
            flush()
    flush()
    return chunks


def load_and_chunk(data_dir: Path = DATA_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() in {".md", ".txt", ".pdf"}:
            chunks.extend(chunk_document(path.name, _read_file(path)))
    return chunks


def data_fingerprint(data_dir: Path = DATA_DIR) -> str:
    h = hashlib.sha256()
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() in {".md", ".txt", ".pdf"}:
            h.update(path.name.encode())
            h.update(path.read_bytes())
    h.update(f"{TARGET_WORDS}-{MAX_WORDS}-{OVERLAP_WORDS}-{EMBED_MODEL}".encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------
# 2. Tokenizer for BM25
# ----------------------------------------------------------------------------
STOPWORDS = set("""
a an the and or but if then so of to in on at by for with from about as into over
is am are was were be been being do does did have has had i me my mine you your
yours we our us he she it its they them their what which who whom whose when where
why how this that these those there here not no can could would should will shall
may might must just also very too than such any all some more most other only own
same s t don doesn didn isn wasn aren weren hasn haven hadn won wouldn shouldn
couldn ll re ve d m o y up out off again further once tell please
""".split())


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9+#]+(?:[-'][a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


# ----------------------------------------------------------------------------
# 3. Retrievers
# ----------------------------------------------------------------------------
@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    detail: str = ""   # extra info (e.g. the BM25/dense ranks behind an RRF score)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder):
        assert len(chunks) == len(embeddings), "chunks.json and embeddings.npy are out of sync"
        self.chunks = chunks
        self.embedder = embedder
        # Normalise once so cosine similarity == dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.emb = embeddings / np.clip(norms, 1e-12, None)
        self.bm25_index = BM25Okapi([tokenize(c.text) for c in chunks])

    # --- full score vectors -------------------------------------------------
    def bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25_index.get_scores(tokenize(query)), dtype=float)

    def dense_scores(self, query: str) -> np.ndarray:
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        return self.emb @ q            # cosine similarity with numpy, no vector DB

    # --- ranked results -----------------------------------------------------
    def _top(self, scores: np.ndarray, k: int, detail=None) -> list[Hit]:
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[i], float(scores[i]), r + 1, detail(i) if detail else "")
                for r, i in enumerate(order)]

    def bm25(self, query: str, k: int = 5) -> list[Hit]:
        # Chunks with score 0 share no query term, so they are not real BM25 matches.
        return [h for h in self._top(self.bm25_scores(query), k) if h.score > 0]

    def dense(self, query: str, k: int = 5) -> list[Hit]:
        return self._top(self.dense_scores(query), k)

    def hybrid(self, query: str, k: int = 5) -> list[Hit]:
        """Reciprocal Rank Fusion: score(d) = sum over retrievers of 1 / (RRF_K + rank(d))."""
        bm = self.bm25_scores(query)
        de = self.dense_scores(query)
        bm_rank = _ranks(bm)
        de_rank = _ranks(de)
        # A chunk with zero BM25 score shares no query terms; it gets no BM25 vote.
        bm_part = np.where(bm > 0, 1.0 / (RRF_K + bm_rank), 0.0)
        de_part = 1.0 / (RRF_K + de_rank)
        fused = bm_part + de_part
        return self._top(
            fused, k,
            detail=lambda i: f"BM25 rank {bm_rank[i] if bm[i] > 0 else '-'}, dense rank {de_rank[i]}",
        )

    def weighted(self, query: str, k: int = 5, alpha: float = 0.5) -> list[Hit]:
        """Alternative fusion (used only for comparison in the report):
        alpha * minmax(dense) + (1 - alpha) * minmax(bm25)."""
        fused = alpha * _minmax(self.dense_scores(query)) + (1 - alpha) * _minmax(self.bm25_scores(query))
        return self._top(fused, k)

    def retrieve(self, query: str, mode: str, k: int = 5) -> list[Hit]:
        mode = mode.lower()
        if mode.startswith("bm25"):
            return self.bm25(query, k)
        if mode.startswith("dense"):
            return self.dense(query, k)
        return self.hybrid(query, k)


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every item (rank 1 = highest score)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


# ----------------------------------------------------------------------------
# Index persistence
# ----------------------------------------------------------------------------
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def build_index(embedder=None) -> tuple[list[Chunk], np.ndarray]:
    embedder = embedder or load_embedder()
    chunks = load_and_chunk()
    emb = embedder.encode([c.text for c in chunks], normalize_embeddings=True,
                          batch_size=32, show_progress_bar=True).astype(np.float32)
    np.save(EMB_PATH, emb)
    CHUNKS_PATH.write_text(json.dumps(
        {"fingerprint": data_fingerprint(), "model": EMBED_MODEL,
         "chunks": [asdict(c) for c in chunks]}, indent=1, ensure_ascii=False), encoding="utf-8")
    return chunks, emb


def load_index(embedder=None) -> HybridRetriever:
    """Load precomputed chunks + embeddings; rebuild only if the data files changed."""
    embedder = embedder or load_embedder()
    if CHUNKS_PATH.exists() and EMB_PATH.exists():
        meta = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == data_fingerprint():
            chunks = [Chunk(**c) for c in meta["chunks"]]
            return HybridRetriever(chunks, np.load(EMB_PATH), embedder)
        print("[rag] data/ changed since embeddings.npy was built -> rebuilding index")
    chunks, emb = build_index(embedder)
    return HybridRetriever(chunks, emb, embedder)
