"""
Digital Clone - Hybrid RAG chatbot (BM25 + dense embeddings + Reciprocal Rank Fusion).

Retrieval code lives in rag.py; this file holds the persona prompt, the LLM call
and the Gradio UI.

LLM access (keys stored ONLY as Space secrets, never in code), tried in order:
  * GEMINI_API_KEY -> Google Gemini free tier (OpenAI-compatible endpoint)
  * HF_TOKEN       -> Hugging Face Inference Providers
  * if every backend fails -> templated answer from the top retrieved chunk
"""

from __future__ import annotations

import html
import os

import gradio as gr
import httpx

from rag import Hit, load_index

# ----------------------------------------------------------------------------
# Persona  (edit these three lines to match yourself)
# ----------------------------------------------------------------------------
OWNER_NAME = os.environ.get("CLONE_NAME", "Usman")
OWNER_DESC = "a BSCS student"
OWNER_TONE = "friendly, casual and to the point, like a student chatting with a classmate"

BANNER = "⚠️ This is an AI clone, not the real person."
NOT_FOUND = "I don't have that in my notes."
TAG = "\n\n<sub>"   # start of the small "mode / backend" footer added under each answer

PERSONA_PROMPT = f"""You are the AI clone of {OWNER_NAME}, {OWNER_DESC}.
Speak in the first person ("I", "my"), in a {OWNER_TONE} tone. Keep answers short: 1 to 5 sentences.

Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own documents.
2. If the context does not contain the answer, say exactly: "{NOT_FOUND}" Do not guess or invent facts about me.
3. Never share phone numbers, addresses, ID numbers, passwords or other private details.
4. If asked whether you are the real person, say you are an AI clone of {OWNER_NAME}.
5. Do not mention "the context" or "the documents" - just talk like me."""

USER_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}"""

# ----------------------------------------------------------------------------
# Startup: load precomputed chunks + embeddings (rebuilds only if data changed)
# ----------------------------------------------------------------------------
RETRIEVER = load_index()

# Free Hugging Face accounts can host Gradio Spaces on "ZeroGPU" hardware. ZeroGPU
# expects at least one @spaces.GPU function; retrieval itself runs fine on CPU, so
# this is just a tiny no-op. Locally (no `spaces` package) it is skipped.
try:
    import spaces

    @spaces.GPU(duration=5)
    def _zerogpu_probe() -> bool:
        return True
except ImportError:
    pass
print(f"[app] loaded {len(RETRIEVER.chunks)} chunks")


# ----------------------------------------------------------------------------
# LLM backends, tried in order. Keys are read only from environment variables
# (Space secrets) and are never written in the code or the repo.
# ----------------------------------------------------------------------------
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
# Main model first; the others are backups for when Google returns "high demand" (503).
GEMINI_MODELS = os.environ.get(
    "GEMINI_MODELS", "gemini-2.5-flash,gemini-3.1-flash-lite,gemini-flash-lite-latest").split(",")
HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def _gemini(model: str):
    def call(messages: list[dict]) -> str:
        r = httpx.post(GEMINI_URL, timeout=25,
                       headers={"Authorization": f"Bearer {os.environ['GEMINI_API_KEY']}"},
                       json={"model": model, "messages": messages,
                             "temperature": 0.2, "max_tokens": 1024})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    return call


def _call_hf(messages: list[dict]) -> str:
    from huggingface_hub import InferenceClient
    out = InferenceClient(api_key=os.environ["HF_TOKEN"]).chat_completion(
        messages=messages, model=HF_MODEL, max_tokens=350, temperature=0.2)
    return out.choices[0].message.content.strip()


API_BACKENDS = []
if os.environ.get("GEMINI_API_KEY"):
    API_BACKENDS += [(f"Gemini {m.strip()}", _gemini(m.strip())) for m in GEMINI_MODELS if m.strip()]
if os.environ.get("HF_TOKEN"):
    API_BACKENDS.append((f"HF {HF_MODEL}", _call_hf))
DISABLED: set[str] = set()   # backends that returned "out of credits" (402) - skip them afterwards

print("[app] LLM backends: " + (" -> ".join(b[0] for b in API_BACKENDS) or "none (templated answers)"))


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.chunk.chunk_id}]\n{h.chunk.text}" for h in hits)


def _text(content) -> str:
    """Gradio 6 may store message content as a list of {"type": "text", "text": ...} parts."""
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def build_messages(question: str, hits: list[Hit], history: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": PERSONA_PROMPT}]
    # Short conversation memory: last 2 exchanges, so follow-up questions work.
    for m in history[-4:]:
        messages.append({"role": m["role"], "content": _text(m["content"]).split(TAG)[0]})
    messages.append({"role": "user",
                     "content": USER_TEMPLATE.format(context=format_context(hits), question=question)})
    return messages


def generate(question: str, hits: list[Hit], history: list[dict]) -> tuple[str, str]:
    """Returns (answer, name of the backend that produced it)."""
    if not hits:   # e.g. BM25 found no chunk sharing a word with the question
        return NOT_FOUND + " (No matching passages were retrieved in this mode.)", "no retrieval"
    messages = build_messages(question, hits, history)
    for name, call in API_BACKENDS:
        if name in DISABLED:
            continue
        try:
            return call(messages), name
        except Exception as e:
            print(f"[app] {name} failed: {type(e).__name__}: {str(e)[:200]}")
            if "402" in str(e) or "Payment Required" in str(e):
                DISABLED.add(name)
    # Last resort: templated answer, so the app never crashes.
    return ("*(LLM unavailable right now - templated answer.)* Here is what my notes say "
            f"(top source `{hits[0].chunk.chunk_id}`):\n\n> {hits[0].chunk.text[:600]}..."), "template"


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def hits_table(hits: list[Hit]) -> list[list]:
    return [[h.rank, h.chunk.source, h.chunk.chunk_id, round(h.score, 4), h.detail,
             h.chunk.text[:160] + ("..." if len(h.chunk.text) > 160 else "")] for h in hits]


def hits_html(hits: list[Hit]) -> str:
    parts = []
    for h in hits:
        extra = f" &middot; {html.escape(h.detail)}" if h.detail else ""
        parts.append(
            f"<details><summary><b>#{h.rank}</b> <code>{html.escape(h.chunk.chunk_id)}</code> "
            f"&middot; score {h.score:.4f}{extra}</summary>"
            f"<div style='white-space:pre-wrap;padding:6px 0 10px'>{html.escape(h.chunk.text)}</div></details>")
    return "\n".join(parts)


TABLE_HEADERS = ["Rank", "Source file", "Chunk ID", "Score", "Fusion detail", "Preview"]


def answer(question: str, mode: str, k: int, history: list[dict]):
    question = (question or "").strip()
    if not question:
        return history, gr.update(), "", ""
    hits = RETRIEVER.retrieve(question, mode, int(k))
    reply, backend = generate(question, hits, history)
    reply += f"{TAG}mode: {mode} · answered by: {backend}</sub>"
    history = history + [{"role": "user", "content": question},
                         {"role": "assistant", "content": reply}]
    return history, hits_table(hits), hits_html(hits), ""


def compare(question: str, k: int):
    question = (question or "").strip()
    if not question:
        return [], [], []
    k = int(k)
    return (hits_table(RETRIEVER.bm25(question, k)),
            hits_table(RETRIEVER.dense(question, k)),
            hits_table(RETRIEVER.hybrid(question, k)))


EXAMPLES = [
    "Who are you and what do you study?",
    "What are your hobbies?",
    "What is the project you are most proud of?",
    "What advice would you give a first-year student?",
    "What is your phone number?",
]

CSS = """
#banner {background:#fff4e5;border:2px solid #f59e0b;border-radius:10px;padding:10px 16px;
         text-align:center;font-size:1.15em;font-weight:700;color:#7c2d12}
"""

with gr.Blocks(title=f"{OWNER_NAME}'s Digital Clone") as demo:
    gr.HTML(f"<div id='banner'>{BANNER}</div>")
    gr.Markdown(f"# 🤖 {OWNER_NAME}'s Digital Clone\n"
                "Ask me anything about myself. I answer only from documents I wrote about me, "
                "using hybrid retrieval (BM25 + dense embeddings, fused with Reciprocal Rank Fusion).")

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Conversation", height=380)
                question = gr.Textbox(label="Your question", placeholder="e.g. What are you studying?  (press Enter to ask)",
                                      lines=1)
                with gr.Row():
                    ask = gr.Button("Ask", variant="primary")
                    clear = gr.Button("Clear chat")
                gr.Examples(EXAMPLES, inputs=question)
            with gr.Column(scale=1):
                mode = gr.Radio(["Hybrid", "BM25 only", "Dense only"], value="Hybrid",
                                label="Retrieval mode")
                topk = gr.Slider(1, 8, value=4, step=1, label="Top-k chunks")
        gr.Markdown("### 📚 Retrieved chunks (source, chunk ID, score)")
        table = gr.Dataframe(headers=TABLE_HEADERS, wrap=True, interactive=False)
        with gr.Accordion("Full text of retrieved chunks", open=False):
            full = gr.HTML()

        ask.click(answer, [question, mode, topk, chat], [chat, table, full, question])
        question.submit(answer, [question, mode, topk, chat], [chat, table, full, question])
        clear.click(lambda: ([], [], ""), None, [chat, table, full])

    with gr.Tab("Compare retrievers"):
        gr.Markdown("See what **BM25**, **Dense** and **Hybrid (RRF)** each retrieve for the same question.")
        cq = gr.Textbox(label="Question")
        ck = gr.Slider(1, 8, value=3, step=1, label="Top-k")
        cbtn = gr.Button("Compare", variant="primary")
        t_bm = gr.Dataframe(headers=TABLE_HEADERS, label="BM25 only", wrap=True, interactive=False)
        t_de = gr.Dataframe(headers=TABLE_HEADERS, label="Dense only", wrap=True, interactive=False)
        t_hy = gr.Dataframe(headers=TABLE_HEADERS, label="Hybrid (RRF)", wrap=True, interactive=False)
        cbtn.click(compare, [cq, ck], [t_bm, t_de, t_hy])
        cq.submit(compare, [cq, ck], [t_bm, t_de, t_hy])

    gr.Markdown(f"<small>{BANNER} Answers may be incomplete; when my notes don't cover something "
                f"I'll say \"{NOT_FOUND}\"</small>")

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
