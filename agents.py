"""MarineWise AI - the 'brain'.
1) API keys  2) in-memory knowledge base (upload PDFs or Google Drive)  3) hybrid search
4) CrewAI agents. A shared 'state' dict travels through the crew (Orchestrator = run_workflow)."""
import json
import os
import re
import tempfile

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CREW_MODEL = "groq/openai/gpt-oss-120b"  # CrewAI talks to Groq through litellm
GEMINI_MODEL = "gemini-2.5-flash"
MIN_SCORE = 0.50  # below this semantic score -> "Not found in the OEM manuals"
MAKERS = ["CAT", "MTU", "YANMAR", "YAMAHA", "HONDA", "MAN"]
SAVE_DIR = "index"
IMG_DIR = os.path.join(tempfile.gettempdir(), "marinewise_images")
MAX_IMAGES = 300

RULES = ("Use ONLY the manual excerpts or facts you are given. Never invent torque values, "
         "clearances, part numbers or steps. If something is missing, say 'see OEM manual'. "
         "Cite every point as (Document, p.PAGE).")


# ================================================================ keys
def get_key(name: str):
    """Read an API key from Streamlit secrets or environment variables (never hardcoded)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


# ================================================================ knowledge base (in memory)
@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def get_embedder():
    """Small CPU embedding model, loaded once."""
    from fastembed import TextEmbedding
    return TextEmbedding(EMBED_MODEL)


def tokenize(text: str) -> list:
    """Lowercase words/numbers for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def get_kb():
    """Return this user's knowledge base (or None)."""
    return st.session_state.get("kb")


def set_kb(chunks: list, vecs: np.ndarray, save: bool = True) -> None:
    """Build FAISS + BM25 from chunks/vectors and keep them in session memory."""
    from rank_bm25 import BM25Okapi
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    st.session_state["kb"] = {"chunks": chunks, "vecs": vecs, "index": index,
                              "bm25": BM25Okapi([tokenize(c["text"]) for c in chunks])}
    if save:  # optional copy on disk so a restart in the same folder can reload it
        try:
            os.makedirs(SAVE_DIR, exist_ok=True)
            with open(os.path.join(SAVE_DIR, "chunks.json"), "w", encoding="utf-8") as f:
                json.dump(chunks, f)
            np.save(os.path.join(SAVE_DIR, "vecs.npy"), vecs)
        except OSError:
            pass


def load_saved() -> None:
    """If a saved index folder exists and nothing is loaded yet, load it."""
    p = os.path.join(SAVE_DIR, "chunks.json")
    if get_kb() is None and os.path.exists(p) and os.path.exists(os.path.join(SAVE_DIR, "vecs.npy")):
        try:
            with open(p, encoding="utf-8") as f:
                set_kb(json.load(f), np.load(os.path.join(SAVE_DIR, "vecs.npy")), save=False)
        except Exception:
            pass


def detect_maker(text: str) -> str:
    """Guess manufacturer from a path/file name (else OTHER)."""
    up = text.upper()
    for m in MAKERS:
        if re.search(r"(^|[^A-Z])" + m + r"([^A-Z]|$)", up):
            return m
    return "OTHER"


def entries_from_uploads(files: list, maker: str, model: str) -> list:
    """Save uploaded PDFs to a temp folder; return a list of file entries."""
    folder = tempfile.mkdtemp(prefix="upload_")
    out = []
    for f in files:
        path = os.path.join(folder, f.name)
        with open(path, "wb") as fh:
            fh.write(f.getvalue())
        out.append({"path": path, "name": f.name, "drive_path": "Uploaded/" + f.name, "maker": maker,
                    "model": model or os.path.splitext(f.name)[0]})
    return out


def entries_from_drive(url: str, keyword: str, max_files: int) -> list:
    """Download a public Drive folder with gdown (max ~50 files per folder) and list its PDFs."""
    import gdown
    root = tempfile.mkdtemp(prefix="drive_")
    gdown.download_folder(url=url, output=root, quiet=True, remaining_ok=True)
    out = []
    for folder, _, names in os.walk(root):
        for n in sorted(names):
            full = os.path.join(folder, n)
            rel = os.path.relpath(full, root).replace("\\", "/")
            if n.lower().endswith(".pdf") and keyword.upper() in rel.upper():
                model = os.path.basename(folder) if folder != root else "Drive"
                out.append({"path": full, "name": n, "drive_path": "Drive/" + rel,
                            "maker": detect_maker(rel), "model": model})
    return out[:max_files]


def save_image(page, doc, name: str) -> str:
    """Save the biggest picture on a page as a small JPEG (illustration only)."""
    best = None
    for info in page.get_images(full=True):
        if info[2] >= 300 and info[3] >= 300 and (best is None or info[2] * info[3] > best[0]):
            best = (info[2] * info[3], info[0])
    if not best:
        return ""
    try:
        from PIL import Image
        pix = fitz.Pixmap(doc, best[1])
        if pix.alpha:
            pix = fitz.Pixmap(pix, 0)
        if pix.colorspace.n != 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.thumbnail((600, 600))
        os.makedirs(IMG_DIR, exist_ok=True)
        path = os.path.join(IMG_DIR, name + ".jpg")
        img.save(path, "JPEG", quality=60)
        return path
    except Exception:
        return ""


def extract_pdf(e: dict, max_pages: int, chunks: list, img_count: list) -> None:
    """Read one PDF page by page -> ~800-char chunks with 100 overlap (never mixing pages)."""
    doc = fitz.open(e["path"])
    for pno, page in enumerate(doc, start=1):
        if pno > max_pages:
            break
        text = " ".join(page.get_text().split())
        if len(text) < 30:  # scanned page, no text: skipped in MVP
            continue
        img = ""
        if img_count[0] < MAX_IMAGES:
            img = save_image(page, doc, f"{abs(hash(e['path']))}_{pno}")
            img_count[0] += 1 if img else 0
        for start in range(0, len(text), 700):
            piece = text[start:start + 800]
            if start > 0 and len(piece) < 50:
                break
            chunks.append({"manufacturer": e["maker"], "model": e["model"], "doc": e["name"],
                           "page": pno, "path": e["drive_path"], "image": img, "text": piece})


def build_kb(entries: list, max_pages: int, progress=None) -> int:
    """Extract -> chunk -> embed -> FAISS + BM25 (adds to any existing KB). Returns new chunk count."""
    say = progress or (lambda f, t: None)
    chunks, img_count = [], [0]
    for n, e in enumerate(entries):
        say(0.4 * n / len(entries), f"Reading {e['name']}")
        try:
            extract_pdf(e, max_pages, chunks, img_count)
        except Exception:
            say(0.4 * n / len(entries), f"Could not read {e['name']} - skipped")
    if not chunks:
        return 0
    emb, vecs = get_embedder(), []
    for i, v in enumerate(emb.embed([c["text"] for c in chunks], batch_size=32)):
        vecs.append(v)
        if i % 64 == 0:
            say(0.4 + 0.55 * i / len(chunks), f"Embedding {i}/{len(chunks)} chunks")
    vecs = np.array(vecs, dtype="float32")
    faiss.normalize_L2(vecs)
    old = get_kb()
    all_chunks, all_vecs = chunks, vecs
    if old:
        all_chunks, all_vecs = old["chunks"] + chunks, np.vstack([old["vecs"], vecs])
    set_kb(all_chunks, all_vecs)
    say(1.0, "Done")
    return len(chunks)


# ================================================================ hybrid search
def search(query: str, maker: str = "OTHER", k: int = 5):
    """FAISS + BM25 merged with Reciprocal Rank Fusion. Returns (top_chunks, best_semantic_score)."""
    kb = get_kb()
    if kb is None:
        raise RuntimeError("Knowledge base is empty. Build it first in the 📚 Knowledge Base tab.")
    chunks = kb["chunks"]
    allowed = None
    if maker and maker != "OTHER":
        allowed = {i for i, c in enumerate(chunks) if c["manufacturer"] == maker} or None
    vec = np.array(list(get_embedder().query_embed([query])), dtype="float32")
    faiss.normalize_L2(vec)
    scores, ids = kb["index"].search(vec, min(1000, len(chunks)))
    sem = [(int(i), float(s)) for i, s in zip(ids[0], scores[0])
           if i >= 0 and (allowed is None or int(i) in allowed)][:20]
    bm = kb["bm25"].get_scores(tokenize(query))
    kw = [int(i) for i in np.argsort(bm)[::-1][:1000]
          if bm[i] > 0 and (allowed is None or int(i) in allowed)][:20]
    rrf = {}
    for rank, (i, _) in enumerate(sem):
        rrf[i] = rrf.get(i, 0) + 1 / (60 + rank)
    for rank, i in enumerate(kw):
        rrf[i] = rrf.get(i, 0) + 1 / (60 + rank)
    top = sorted(rrf, key=rrf.get, reverse=True)[:k]
    return [chunks[i] for i in top], (sem[0][1] if sem else 0.0)


def cite(c: dict) -> str:
    """Source format: Document | Page | Drive path."""
    return f"{c['doc']} | Page {c['page']} | {c['path']}"


def context_of(hits: list) -> str:
    """Turn chunks into text the agents can read."""
    return "\n\n".join(f"[{c['doc']}, p.{c['page']}]\n{c['text']}" for c in hits)


def as_json(text: str):
    """Pull the first JSON object/array out of an LLM reply."""
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        raise ValueError("The AI did not return valid JSON. Please try again.")
    return json.loads(m.group(0))


# ================================================================ CrewAI core
def run_crew(specs: list) -> str:
    """Run a CrewAI crew. specs = list of (role, goal, backstory, task_description, expected_output).
    Agents work one after another; each task receives earlier tasks' output as context."""
    key = get_key("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it in Streamlit secrets.")
    from crewai import Agent, Crew, LLM, Process, Task
    llm = LLM(model=CREW_MODEL, api_key=key, temperature=0.2)
    agents, tasks = [], []
    for role, goal, backstory, description, expected in specs:
        agent = Agent(role=role, goal=goal, backstory=f"{backstory} {RULES}", llm=llm,
                      allow_delegation=False, verbose=False)
        extra = {"context": list(tasks)} if tasks else {}
        tasks.append(Task(description=description, expected_output=expected, agent=agent, **extra))
        agents.append(agent)
    return Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False).kickoff().raw


# ---- the agents (each returns one spec = role, goal, backstory, task, expected output)
def knowledge_agent(query: str, ctx: str) -> tuple:
    """Agent 1 - Knowledge/RAG: turns retrieved excerpts into cited facts."""
    return ("OEM Knowledge Agent", "Extract only facts from the OEM excerpts that answer the request",
            "A librarian of marine OEM manuals who never adds facts.",
            f"Request: {query}\n\nManual excerpts:\n{ctx}\n\nList every relevant fact as a bullet "
            "ending with (Document, p.PAGE). Quote exact values only if written in the excerpts.",
            "Bullet list of cited facts.")


def troubleshooting_agent(s: dict) -> tuple:
    """Agent 2 - Troubleshooting engineer."""
    return ("Marine Troubleshooting Engineer", "Diagnose engine defects using the cited facts",
            "A senior marine engineer who trusts only the OEM manual.",
            f"Engine: {s['maker']} {s['model']} (serial {s.get('serial') or 'n/a'}). Problem: {s['topic']}.\n"
            "Using ONLY the facts from the OEM Knowledge Agent give: 1) Possible causes 2) Checks "
            "3) Corrective steps 4) Warnings. Short bullets, each with (Document, p.PAGE).",
            "Markdown bullets grouped under the 4 headings, with citations.")


def training_agent(s: dict) -> tuple:
    """Agent 3 - Training designer (returns JSON)."""
    return ("Technical Training Designer", "Build clear technician training from the cited facts",
            "An experienced marine instructor.",
            f"Engine: {s['model']}. Vessel: {s.get('ship', '')}. Topic: {s['topic']}. Level: {s['level']}.\n"
            "Return ONLY JSON: {\"title\":str, \"overview\":str, \"objectives\":[str], "
            "\"steps\":[{\"title\":str, \"points\":[str], \"source\":\"Doc p.X\"}], \"safety\":[str], "
            "\"mistakes\":[str], \"exercise\":str, \"practice\":[{\"q\":str, \"a\":str}]}. "
            "4-7 steps, max 5 short points each. Use only the cited facts.",
            "Valid JSON only.")


def assessment_agent(s: dict) -> tuple:
    """Agent 4 - Assessment examiner (returns JSON)."""
    fmt = {"MCQ": "4 options in 'options' (no letters); 'answer' is the letter A-D",
           "True/False": "'options' is []; 'answer' is True or False",
           "Short answer": "'options' is []; 'answer' is a short model answer"}[s["qtype"]]
    return ("Assessment Examiner", "Write fair technical questions from the cited facts",
            "A marine examiner who writes clear, unambiguous questions.",
            f"Engine: {s['model']}. Topic: {s['topic']}. Write exactly {s['n']} {s['qtype']} questions. "
            "Return ONLY a JSON array of {\"q\":str, \"options\":[str], \"answer\":str, "
            f"\"explanation\":str, \"source\":\"Doc p.X\"}}. {fmt}.",
            "Valid JSON array only.")


def quality_agent(ctx: str, fmt: str) -> tuple:
    """Agent 5 - Document/quality check: removes anything the manual does not support."""
    return ("Quality Checker", "Make sure every statement and every section has a source in the excerpts",
            "A strict OEM quality inspector who deletes unsupported claims.",
            f"Check the previous draft against these excerpts. Delete or fix anything unsupported, "
            f"make sure every section/step has a source, keep the same format.\n\nExcerpts:\n{ctx}", fmt)


SPECIALISTS = {"troubleshoot": troubleshooting_agent, "training": training_agent, "quiz": assessment_agent}
FORMATS = {"troubleshoot": "Markdown bullets with citations.", "training": "Valid JSON only.",
           "quiz": "Valid JSON array only."}


def run_workflow(kind: str, state: dict) -> dict:
    """ORCHESTRATOR. Shared 'state' dict -> retrieve -> Knowledge agent -> Specialist -> Quality check.
    Results are stored back into the state (found, raw, chunks, sources)."""
    query = f"{state['model']} {state['topic']}"
    hits, best = search(query, state.get("maker", "OTHER"), k=state.get("k", 8))
    if (not hits or best < MIN_SCORE) and state.get("maker", "OTHER") != "OTHER":
        hits, best = search(query, "OTHER", k=state.get("k", 8))  # retry without manufacturer filter
    state["chunks"] = hits
    state["found"] = bool(hits) and best >= MIN_SCORE
    if not state["found"]:
        return state
    ctx = context_of(hits)
    state["raw"] = run_crew([knowledge_agent(query, ctx), SPECIALISTS[kind](state),
                             quality_agent(ctx, FORMATS[kind])])
    state["sources"] = sorted({cite(c) for c in hits})
    return state


def guess_maker(model: str) -> str:
    """Manufacturer from the first word of the model text (else OTHER)."""
    first = model.split()[0].upper() if model.split() else ""
    return first if first in MAKERS else "OTHER"


# ================================================================ public functions used by app.py
def troubleshoot(maker: str, model: str, serial: str, defect: str) -> dict:
    """Troubleshooting workflow. found=False -> app offers the online search."""
    s = run_workflow("troubleshoot", {"maker": maker, "model": model, "serial": serial,
                                      "topic": defect, "k": 6})
    return {"found": s["found"], "answer": s.get("raw", ""), "sources": s.get("sources", [])}


def make_training(model: str, ship: str, topic: str, level: str):
    """Training workflow. Returns content dict (with images + sources) or None if not found."""
    s = run_workflow("training", {"maker": guess_maker(model), "model": model, "ship": ship,
                                  "topic": topic, "level": level, "k": 10})
    if not s["found"]:
        return None
    data = as_json(s["raw"])
    imgs = [c["image"] for c in s["chunks"] if c.get("image") and os.path.exists(c["image"])]
    for n, step in enumerate(data.get("steps", [])):  # pictures are illustrations only
        step["image"] = imgs[n] if n < len(imgs) else ""
    data["subtitle"] = f"{model} | {ship} | {level}"
    data["sources"] = s["sources"]
    return data


def make_quiz(model: str, topic: str, qtype: str, n: int):
    """Assessment workflow. Returns a list of questions with answer key, or None if not found."""
    s = run_workflow("quiz", {"maker": guess_maker(model), "model": model, "topic": topic,
                              "qtype": qtype, "n": n, "k": 10})
    if not s["found"]:
        return None
    quiz = as_json(s["raw"])[:n]
    for item in quiz:
        item["type"] = qtype
    return quiz


def web_search(maker: str, model: str, defect: str) -> dict:
    """Research agent (Tavily) - only called after the user clicks YES."""
    key = get_key("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is missing. Add it in Streamlit secrets.")
    from tavily import TavilyClient
    items = TavilyClient(api_key=key).search(f"{maker} {model} marine engine {defect}",
                                             max_results=5).get("results", [])
    if not items:
        return {"answer": "No useful web results found.", "urls": []}
    text = "\n\n".join(f"[{r['url']}]\n{r['content']}" for r in items)
    desc = (f"Problem: {maker} {model} {defect}\n\nWeb excerpts:\n{text}\n\nUse ONLY the web excerpts. "
            "Cite the URL for each point. State clearly this is NOT from the OEM manual and must be verified.")
    answer = run_crew([("Web Research Analyst", "Find careful, sourced help online",
                        "A cautious researcher who never guesses.", desc,
                        "Short bullets with URLs and a verification warning.")])
    return {"answer": answer, "urls": [r["url"] for r in items]}


def read_answer_sheet(img_bytes: bytes, mime: str, n: int) -> dict:
    """Gemini vision reads the technician's handwritten answers."""
    key = get_key("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it in Streamlit secrets.")
    from google import genai
    from google.genai import types
    prompt = (f"This is a photo of a completed quiz answer sheet with {n} questions. Read the "
              "technician's answer for each question number. Return ONLY JSON like "
              "{\"1\":\"B\",\"2\":\"True\",\"3\":\"short text\"}. Use \"\" if blank. Do not guess.")
    r = genai.Client(api_key=key).models.generate_content(
        model=GEMINI_MODEL, contents=[types.Part.from_bytes(data=img_bytes, mime_type=mime), prompt])
    return as_json(r.text)


def grade(quiz: list, given: dict) -> dict:
    """MCQ/True-False matched in code; short answers judged by a CrewAI grader agent."""
    ok, shorts = {}, {}
    for i, q in enumerate(quiz, start=1):
        g = str(given.get(str(i), "")).strip()
        if q["type"] == "MCQ":
            ok[i] = g != "" and g[:1].upper() == str(q["answer"]).strip()[:1].upper()
        elif q["type"] == "True/False":
            ok[i] = g != "" and g[:1].lower() == str(q["answer"]).strip()[:1].lower()
        else:
            shorts[i] = {"question": q["q"], "key": q["answer"], "given": g}
    if shorts:
        desc = ("Mark each answer true if it matches the key in meaning, else false. "
                "Return ONLY JSON like {\"3\":true}.\n\n" + json.dumps(shorts))
        verdict = as_json(run_crew([("Strict Marine Grader", "Mark technician answers fairly",
                                     "A strict marine instructor.", desc, "JSON only.")]))
        for i in shorts:
            ok[i] = bool(verdict.get(str(i), False)) and shorts[i]["given"] != ""
    correct = sum(ok.values())
    return {"ok": ok, "correct": correct, "total": len(quiz),
            "percent": round(100 * correct / max(len(quiz), 1), 1),
            "wrong": [quiz[i - 1]["q"] for i, v in ok.items() if not v]}
