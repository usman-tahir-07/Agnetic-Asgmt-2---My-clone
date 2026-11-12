---
title: Usman Digital Clone
emoji: 🤖
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
short_description: Hybrid RAG (BM25 + dense + RRF) AI clone of Usman
---

# 🤖 Usman's Digital Clone — Hybrid RAG

> **⚠️ This is an AI clone, not the real person.**

A chatbot that answers questions *as me* (Usman Tahir, BSCS student), using only documents I wrote about
myself. It retrieves passages with **BM25** (keyword) and **dense embeddings** (`all-MiniLM-L6-v2`, cosine
similarity in numpy, no vector DB), fuses them with **Reciprocal Rank Fusion**, and sends the top chunks to an
LLM with a first-person persona prompt that must say *"I don't have that in my notes."* when the answer isn't there.

## How it works

| Step | Implementation |
|---|---|
| Load & chunk | `rag.py`: paragraph-aware packing into ~250-word chunks (max 300) with a 40-word overlap. Every chunk keeps its `source` file name and a `chunk_id` (e.g. `faq.md#3`). 9 documents → 24 chunks (avg 229 words). |
| BM25 | `rank_bm25.BM25Okapi` over lower-cased, stop-word-filtered tokens; returns top-k chunks with scores (chunks with score 0 share no query word and are not returned). |
| Dense | `sentence-transformers/all-MiniLM-L6-v2`; embeddings precomputed once by `build_index.py` and saved to `embeddings.npy` (+ `chunks.json`); cosine similarity = dot product of normalised vectors in numpy. |
| Hybrid | **Reciprocal Rank Fusion**: `score(d) = Σ 1 / (60 + rank_i(d))` over the BM25 and dense rankings (a chunk with BM25 score 0 gets no BM25 vote). The UI shows each fused chunk's BM25 rank and dense rank. |
| Generate | Google **Gemini** free tier (`gemini-2.5-flash`, falling back to `gemini-3.1-flash-lite` / `gemini-flash-lite-latest` on rate limits), then the HF Inference API (`HF_TOKEN`). Keys are **Space secrets** only. Temperature 0.2, top-4 chunks, last 2 exchanges kept as short conversation memory. |
| UI | Gradio: AI-clone banner, question box, answer, mode selector (Hybrid / BM25 only / Dense only), top-k slider, table of retrieved chunks with source + chunk ID + score, full-text panel, and a **Compare retrievers** tab that shows all three modes side by side. |

Files: `app.py` (persona prompt, LLM, UI) · `rag.py` (chunking, BM25, dense, RRF, weighted fusion) ·
`build_index.py` (precompute embeddings) · `evaluate.py` (hit@3 + report table) · `data/` (knowledge base) ·
`chunks.json`, `embeddings.npy` (precomputed index).

---

# Report

## 1. Knowledge base and why

Nine first-person Markdown documents (~4,900 words, about 10 pages), all written about me:

| File | Type | Why it's there |
|---|---|---|
| `bio.md` | factual | who I am, studies, interests, privacy statement |
| `cv.md` | factual | skills with honest levels, tools, projects, languages |
| `faq.md` | **first-person FAQ (28 Q&As)** | the answers in my own casual voice ("ok so", "yaar"); includes honest "I don't have that in my notes" answers |
| `projects-casetrack.md` | long-form | write-up of my DBMS project **CaseTrack** plus SQL and EMU8086 coursework |
| `essay-learning-computing.md` | long-form | essay on what learning computing has taught me |
| `opinions-and-hobbies.md` | personality / opinion | travel, mountains, photography, values, 5 hot takes |
| `study-habits.md` | how I work | quiz prep, coding setup, AI use, debugging approach |
| `advice-and-goals.md` | advice | advice for juniors, short/long-term goals |
| `study-notes.md` | study notes | my revision notes on DBMS/T-SQL, EMU8086 and PDC |

Rare keywords for BM25: *CaseTrack*, *EMU8086*, *SSMS*, *Flynn's taxonomy*, *DCL*. Plain descriptive
passages (hobbies, study habits) let dense search show its strength on paraphrased questions.
I only included facts I could confirm. Details I haven't written down (university name, semester, favourite
food/films, internships) were left out on purpose, so the clone says "I don't have that in my notes" instead of
inventing them. No phone number, address, ID numbers or private details about other people are in the Space.

## 2. Five test questions across the three modes (top-3 chunks, k = 3)

| # | Question | BM25 result | Dense result | Hybrid (RRF) result | Best mode and why |
|---|---|---|---|---|---|
| Q1 | **"Tell me about Flynn's taxonomy"** *(exact rare keyword)* | ✅ `study-notes#3` (5.56); then no other matches | ❌ `cv#1` (0.22), `faq#0`, `faq#2`: generic "studies" chunks; the PDC notes are not in the top 3 | ✅ `study-notes#3` first (BM25 rank 1), then dense's picks | **BM25 / Hybrid.** "Flynn" is a rare exact term with high IDF. MiniLM has no useful embedding for this name, so the dense ranking is generic. RRF keeps BM25's single strong hit at #1. |
| Q2 | **"What helps you unwind?"** *(paraphrase, no shared words with the answer)* | ❌ nothing: no chunk contains "helps" or "unwind" | ✅ `opinions-and-hobbies#0` (0.28): travel to the mountains is "the best reset for me" | ✅ same as dense (only dense votes) | **Dense / Hybrid.** The answer says "best reset", "free time", "mountains", none of which are query words. Only semantic similarity links *unwind* ↔ *reset/free time*. Clone: *"Travelling to scenic places, especially the mountains, really helps me unwind…"* |
| Q3 | "Which addressing modes did you use in EMU8086?" | ⚠️ `projects-casetrack#3` (3.75), `cv#0`, `cv#1`: they mention EMU8086/addressing modes but don't list them | ✅ `study-notes#2` (0.33), the notes that list immediate/register/direct/indirect/indexed modes, then `projects-casetrack#3` | ✅ `projects-casetrack#3`, `study-notes#2`, `faq#1`: both the experience and the actual list | **Hybrid.** BM25 finds where I *talk about* EMU8086; dense finds the chunk that *explains* the modes. RRF puts both in the top 2, so the answer covers what I did and what the modes are. The fusion visibly changes the ranking (BM25 #1 + dense #1 interleaved). |
| Q4 | "What is the difference between a login and a user in SQL Server?" | ✅ `study-notes#1` (9.01), `study-notes#0` (8.98) | ✅ `study-notes#1` (0.44), `cv#1`, `study-notes#0` | ✅ `study-notes#1`, `study-notes#0`, `cv#0` | **Hybrid ≈ BM25.** Both retrievers agree, and RRF rewards the agreement: both notes chunks rank 1–2. BM25 is very confident here because "login", "user" and "SQL Server" appear literally. |
| Q5 | "How do you get ready for tests?" | ⚠️ `advice-and-goals#0` (1.06), `essay#1`, `faq#2`: weak matches on "get"/"ready" | ✅ `study-habits#0` (0.55): "Preparing for quizzes and exams" | ⚠️ `essay#1`, `advice-and-goals#0`, `faq#2`; `study-habits#0` falls to **rank 9** | **Dense.** "tests" ≠ "quizzes/exams" lexically, so BM25 only matched filler words. RRF gives those noisy BM25 ranks the same weight as dense rank 1, and `study-habits#0` (no BM25 vote) drops out. The clone still answered well because `faq#2` (Q16, quiz prep) was in the hybrid top 3. |

**hit@3 on a 16-question hand-labelled set (`evaluate.py`):**
BM25 **12/16** · Dense **14/16** · Hybrid RRF **15/16** · Weighted fusion (α = 0.5) **15/16**.
Hybrid was never worse than the better single retriever except on Q5-style questions, and it fixed BM25's paraphrase
misses and dense's rare-keyword misses.

**Other fusion method.** I implemented RRF in the app. `rag.py` also has the alternative
`alpha * minmax(dense) + (1 - alpha) * minmax(bm25)`, which I used for comparison. It scored the same hit@3 (15/16).
It uses the actual score gaps, so BM25's huge score on Q4 counts for more, but it depends on choosing α and on
min-max normalisation, which is unstable when all BM25 scores are 0. RRF only needs ranks and has no tuning.

## 3. Failure case

**Question:** "What would you tell a newcomer to the field?"
**Clone (hybrid):** *"I'd say focus on learning useful skills rather than just collecting definitions. It's better to
make steady progress every week… most of what I know about debugging came from fixing my own mistakes!"*

This is a weak answer. It's built from my *values* list in `opinions-and-hobbies.md` and `bio.md#1`, not from my
real advice to juniors in `advice-and-goals.md` (practise regularly, build small projects, read error messages,
ask early, don't compare yourself). All modes missed the right chunk:

- **BM25** found nothing. The advice doc says "first-year", "junior" and "starting computer science", never
  "newcomer" or "field".
- **Dense** ranked `advice-and-goals#0` only **6th** (0.19 vs 0.27 for `bio#1`). The phrase "newcomer to the
  field" is vague, and the advice chunk is a long bullet list whose embedding is averaged over many different
  tips, so it isn't strongly "about" newcomers. The same question phrased as "What advice would you give a junior
  student?" puts `advice-and-goals#0` at rank 1 (0.49).
- **Hybrid** can't fix a chunk that neither retriever ranks highly. RRF only reorders what the retrievers return.
- The LLM then did what it was told: it answered from the context it got, which happened to be related but
  not the best. It didn't invent facts, but the answer is weaker than it should be.

## 4. Improvement I would make next

**Contextual chunk headers + a cross-encoder reranker.** First, prepend each chunk's document title and section
heading (e.g. *"Advice for first-year / junior / new CS students"*) before embedding and BM25 indexing. That gives
both retrievers the topic words that body text leaves implicit, which would fix the failure above. Second,
retrieve the top ~15 hybrid candidates and rerank them with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
It reads the question and the chunk together, so it would also push `study-habits#0` back to the top for Q5.
I'd use `evaluate.py`'s hit@3 to check that it actually helps.

---

### Running locally
```bash
pip install -r requirements.txt
python build_index.py      # only needed after editing data/
GEMINI_API_KEY=... python app.py
```
