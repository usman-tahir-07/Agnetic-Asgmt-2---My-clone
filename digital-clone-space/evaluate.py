"""Optional extra: hand-labelled evaluation set with hit@3 for each retrieval mode,
plus a side-by-side dump used to fill the report table.

    python evaluate.py

Each question is labelled with the source file(s) that contain the answer.
A 'hit' means at least one of the top-3 chunks comes from a correct file.
"""
from rag import load_index

# (question, set of files that contain the answer)  -- edit to match data/
EVAL_SET: list[tuple[str, set[str]]] = [
    ("What is CaseTrack?", {"projects-casetrack.md", "faq.md", "cv.md"}),
    ("Tell me about Flynn's taxonomy", {"study-notes.md"}),
    ("What is the difference between a login and a user in SQL Server?", {"study-notes.md"}),
    ("Which addressing modes did you use in EMU8086?", {"study-notes.md"}),
    ("What helps you unwind?", {"opinions-and-hobbies.md", "bio.md", "faq.md"}),
    ("How do you get ready for tests?", {"study-habits.md", "faq.md"}),
    ("What would you tell a newcomer to the field?", {"advice-and-goals.md", "faq.md"}),
    ("What was the hardest bug you faced?", {"projects-casetrack.md", "faq.md", "essay-learning-computing.md"}),
    ("Do you use ChatGPT?", {"study-habits.md", "faq.md", "opinions-and-hobbies.md"}),
    ("What are your career goals?", {"advice-and-goals.md", "faq.md"}),
    ("What languages do you speak?", {"bio.md", "cv.md", "faq.md"}),
    ("Do you prefer memorising or understanding?", {"opinions-and-hobbies.md", "faq.md"}),
    ("Where do you like to go on holiday?", {"opinions-and-hobbies.md", "bio.md", "faq.md"}),
    ("What laptop and software do you code on?", {"study-habits.md", "faq.md", "cv.md"}),
    ("Which subject do you find hardest?", {"faq.md", "study-notes.md"}),
    ("How do you fix errors in your code?", {"study-habits.md", "faq.md", "essay-learning-computing.md"}),
]

REPORT_QUESTIONS: list[str] = [
    "Tell me about Flynn's taxonomy",
    "What helps you unwind?",
    "Which addressing modes did you use in EMU8086?",
    "What is the difference between a login and a user in SQL Server?",
    "How do you get ready for tests?",
    "What would you tell a newcomer to the field?",
]

K = 3


def main():
    r = load_index()
    if EVAL_SET:
        modes = {"BM25": r.bm25, "Dense": r.dense, "Hybrid (RRF)": r.hybrid,
                 "Weighted a=0.5": r.weighted}
        hits = {m: 0 for m in modes}
        print(f"{'question':<60} " + " ".join(f"{m[:8]:>9}" for m in modes))
        for q, gold in EVAL_SET:
            row = []
            for m, fn in modes.items():
                ok = any(h.chunk.source in gold for h in fn(q, K))
                hits[m] += ok
                row.append("hit" if ok else "MISS")
            print(f"{q[:58]:<60} " + " ".join(f"{x:>9}" for x in row))
        print("\nhit@3: " + ", ".join(f"{m} {hits[m]}/{len(EVAL_SET)}" for m in modes))

    for q in REPORT_QUESTIONS:
        print(f"\n=== {q}")
        for name, fn in (("BM25", r.bm25), ("Dense", r.dense), ("Hybrid", r.hybrid)):
            print(f"  {name:<7}" + " | ".join(f"{h.chunk.chunk_id} ({h.score:.3f})" for h in fn(q, K)))


if __name__ == "__main__":
    main()
