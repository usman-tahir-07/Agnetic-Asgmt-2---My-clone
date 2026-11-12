"""Run locally once (and again whenever you edit data/) to precompute embeddings.

    python build_index.py

Writes chunks.json (chunk text + source file + chunk ID) and embeddings.npy.
"""
from collections import Counter

from rag import build_index

chunks, emb = build_index()
print(f"\n{len(chunks)} chunks, embeddings shape {emb.shape}")
for src, n in sorted(Counter(c.source for c in chunks).items()):
    print(f"  {src:<28} {n} chunks")
lens = [len(c.text.split()) for c in chunks]
print(f"words/chunk: min {min(lens)}, avg {sum(lens)/len(lens):.0f}, max {max(lens)}")
