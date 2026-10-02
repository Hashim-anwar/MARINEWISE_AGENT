"""MarineWise AI - the 'brain': API keys, hybrid search (FAISS + BM25) and the CrewAI agents."""
import json
import os
import re

import faiss
import numpy as np
import streamlit as st

INDEX_DIR = "index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CREW_MODEL = "groq/openai/gpt-oss-120b"  # CrewAI talks to Groq through litellm
GEMINI_MODEL = "gemini-2.5-flash"
MIN_SCORE = 0.50  # below this semantic score we say "Not found in the OEM manuals"

RULES = ("Use ONLY the manual excerpts provided. Never invent torque values, clearances, "
         "part numbers or steps. If the excerpts do not contain something, say so. "
         "Cite every point as (Document, p.PAGE).")


# ---------------------------------------------------------------- keys + LLM
def get_key(name: str):
    """Read an API key from Streamlit secrets or environment variables (never hardcoded)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def run_crew(specs: list) -> str:
    """Run a CrewAI crew. specs = list of (role, goal, backstory, task_description, expected_output).
    Agents work one after another; each task receives the earlier tasks' output as context."""
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
    result = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False).kickoff()
    return result.raw


def reviewer(excerpts: str, fmt: str) -> tuple:
    """Second agent: removes anything the manual excerpts do not support."""
    return ("Quality Reviewer", "Make sure every statement is supported by the manual excerpts",
            "A strict OEM quality inspector who deletes unsupported claims.",
            f"Check the previous draft against these excerpts and delete or fix anything not "
            f"supported. Keep the same format.\n\nExcerpts:\n{excerpts}", fmt)


def as_json(text: str):
    """Pull the first JSON object/array out of an LLM reply."""
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        raise ValueError("The AI did not return valid JSON. Please try again.")
    return json.loads(m.group(0))


# ---------------------------------------------------------------- knowledge base
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_kb():
    """Load the saved FAISS index + chunks, and build BM25 once. Returns None if no index."""
    path = os.path.join(INDEX_DIR, "faiss.index")
    if not os.path.exists(path):
        return None
    from fastembed import TextEmbedding
    from rank_bm25 import BM25Okapi
    with open(os.path.join(INDEX_DIR, "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    return {"index": faiss.read_index(path), "chunks": chunks,
            "bm25": BM25Okapi([tokenize(c["text"]) for c in chunks]),
            "emb": TextEmbedding(EMBED_MODEL)}


def tokenize(text: str) -> list:
    """Lowercase words/numbers for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def search(query: str, maker: str = "OTHER", k: int = 5):
    """Hybrid search: FAISS + BM25 merged with Reciprocal Rank Fusion.
    Returns (top_chunks, best_semantic_score)."""
    kb = load_kb()
    if kb is None:
        return [], 0.0
    chunks = kb["chunks"]
    allowed = None
    if maker and maker != "OTHER":
        allowed = {i for i, c in enumerate(chunks) if c["manufacturer"] == maker} or None
    vec = np.array(list(kb["emb"].query_embed([query])), dtype="float32")
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
    """Format a source as: Document | Page | Google Drive path."""
    return f"{c['doc']} | Page {c['page']} | {c['path']}"


def context_of(hits: list) -> str:
    """Turn chunks into a text block the LLM can read."""
    return "\n\n".join(f"[{c['doc']}, p.{c['page']}]\n{c['text']}" for c in hits)


# ---------------------------------------------------------------- agent 1: troubleshooting
def troubleshoot(maker: str, model: str, serial: str, defect: str) -> dict:
    """Answer a defect/alarm from the OEM manuals only. found=False if evidence is weak."""
    hits, best = search(f"{maker} {model} {defect}", maker, k=6)
    if not hits or best < MIN_SCORE:
        return {"found": False, "sources": []}
    prompt = (f"Engine: {maker} {model} (serial {serial or 'n/a'})\nProblem: {defect}\n\n"
              f"Manual excerpts:\n{context_of(hits)}\n\n"
              "Give: 1) Possible causes 2) Checks 3) Corrective steps 4) Warnings. "
              "Use short bullet points, each with its (Document, p.PAGE) citation.")
    ctx = context_of(hits)
    fmt = "Bullet points grouped as Causes, Checks, Corrective steps, Warnings, each with a citation."
    answer = run_crew([
        ("Marine Troubleshooting Engineer", "Diagnose engine defects from OEM manuals",
         "A senior marine engineer who only trusts the OEM manual.", prompt, fmt),
        reviewer(ctx, fmt)])
    return {"found": True, "answer": answer, "sources": [cite(c) for c in hits]}


def web_search(maker: str, model: str, defect: str) -> dict:
    """Online search with Tavily (only after the user clicks YES). Answers from results only."""
    key = get_key("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is missing. Add it in Streamlit secrets.")
    from tavily import TavilyClient
    res = TavilyClient(api_key=key).search(f"{maker} {model} marine engine {defect}", max_results=5)
    items = res.get("results", [])
    if not items:
        return {"answer": "No useful web results found.", "urls": []}
    text = "\n\n".join(f"[{r['url']}]\n{r['content']}" for r in items)
    desc = (f"Problem: {maker} {model} {defect}\n\nWeb excerpts:\n{text}\n\n"
            "Use ONLY the web excerpts. Cite the URL for each point. State clearly that this is "
            "NOT from the OEM manual and must be verified.")
    answer = run_crew([("Web Research Analyst", "Find careful, sourced help online",
                        "A cautious researcher who never guesses.", desc,
                        "Short bullet points with URLs and a verification warning.")])
    return {"answer": answer, "urls": [r["url"] for r in items]}


# ---------------------------------------------------------------- agent 2: training
def make_training(model: str, ship: str, topic: str, level: str):
    """Build structured training content from OEM manuals. Returns None if not found."""
    hits, best = search(f"{model} {topic}", maker=model.split()[0].upper(), k=10)
    if not hits or best < MIN_SCORE:
        hits, best = search(f"{model} {topic}", "OTHER", k=10)
    if not hits or best < MIN_SCORE:
        return None
    prompt = (f"Engine: {model}. Vessel: {ship}. Topic: {topic}. Level: {level}.\n\n"
              f"Manual excerpts:\n{context_of(hits)}\n\n"
              "Return ONLY JSON: {\"title\":str, \"overview\":str, \"objectives\":[str], "
              "\"steps\":[{\"title\":str, \"points\":[str], \"source\":\"Doc p.X\"}], "
              "\"safety\":[str], \"mistakes\":[str], \"exercise\":str, "
              "\"practice\":[{\"q\":str, \"a\":str}]}. 4-7 steps, max 5 points each, short sentences. "
              "If a value is not in the excerpts write 'see OEM manual'.")
    fmt = "Valid JSON only, same keys as requested."
    data = as_json(run_crew([
        ("Technical Training Designer", "Build clear technician training from OEM manuals",
         "An experienced marine instructor.", prompt, fmt),
        reviewer(context_of(hits), fmt)]))
    imgs = [c["image"] for c in hits if c.get("image") and os.path.exists(c["image"])]
    for n, step in enumerate(data.get("steps", [])):  # pictures are for illustration only
        step["image"] = imgs[n] if n < len(imgs) else ""
    data["subtitle"] = f"{model} | {ship} | {level}"
    data["sources"] = sorted({cite(c) for c in hits})
    return data


# ---------------------------------------------------------------- agent 3: assessment
def make_quiz(model: str, topic: str, qtype: str, n: int):
    """Create n questions with an answer key from the manuals. Returns None if not found."""
    hits, best = search(f"{model} {topic}", maker=model.split()[0].upper(), k=10)
    if not hits or best < MIN_SCORE:
        hits, best = search(f"{model} {topic}", "OTHER", k=10)
    if not hits or best < MIN_SCORE:
        return None
    fmt = {"MCQ": "4 options in 'options' (no letters); 'answer' is the letter A-D",
           "True/False": "'options' is []; 'answer' is True or False",
           "Short answer": "'options' is []; 'answer' is a short model answer"}[qtype]
    prompt = (f"Engine: {model}. Topic: {topic}.\n\nManual excerpts:\n{context_of(hits)}\n\n"
              f"Write exactly {n} {qtype} questions. Return ONLY a JSON array of "
              "{\"q\":str, \"options\":[str], \"answer\":str, \"explanation\":str, "
              f"\"source\":\"Doc p.X\"}}. {fmt}.")
    fmt = "Valid JSON array only, same keys as requested."
    quiz = as_json(run_crew([
        ("Assessment Examiner", "Write fair technical questions from OEM manuals",
         "A marine examiner who writes clear, unambiguous questions.", prompt, fmt),
        reviewer(context_of(hits), fmt)]))[:n]
    for item in quiz:
        item["type"] = qtype
    return quiz


def read_answer_sheet(img_bytes: bytes, mime: str, n: int) -> dict:
    """Use Gemini vision to read the technician's handwritten answers."""
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
    """Compare answers with the key. MCQ/TF are matched; short answers are judged by the LLM."""
    ok = {}
    shorts = {}
    for i, q in enumerate(quiz, start=1):
        g = str(given.get(str(i), "")).strip()
        if q["type"] == "MCQ":
            ok[i] = g[:1].upper() == str(q["answer"]).strip()[:1].upper() and g != ""
        elif q["type"] == "True/False":
            ok[i] = g[:1].lower() == str(q["answer"]).strip()[:1].lower() and g != ""
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
