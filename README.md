# ⚓ MarineWise AI (simple MVP, CrewAI)

Knowledge → Troubleshoot → Train → Assess → Score → Retrain.

## Your files (only 5)
| File | What it does |
|---|---|
| `app.py` | The screen (5 tabs). UI only. |
| `agents.py` | Keys, in-app knowledge base (upload or Drive), FAISS + BM25 search, **CrewAI agents** |
| `documents.py` | Makes PPTX, DOCX, PDF and Quiz PDF |
| `requirements.txt` | Libraries |
| `README.md` | This guide |

Delete the old `build_index.py` - it is no longer needed.

## How the agent flow works
All workflows share ONE Python dict called `state` (engine, topic, level, retrieved chunks, answer, sources).
`run_workflow()` in `agents.py` is the Orchestrator: it fills the state and runs the crew in order.

```
 User (tab) ──► state{maker, model, topic, level...}
                  │
      ┌───────────▼───────────┐
      │ Hybrid search (FAISS+BM25, RRF) │──► weak evidence? ─► "Not found in OEM manuals"
      └───────────┬───────────┘                               └► [Search online?] ─► Research agent (Tavily)
                  ▼  state.chunks
      CrewAI crew (sequential, each task gets the previous output)
        1. OEM Knowledge Agent   -> cited facts
        2. Specialist agent      -> Troubleshooting Engineer | Training Designer | Assessment Examiner
        3. Quality Checker       -> removes unsupported claims, checks sources
                  ▼  state.raw + state.sources
      documents.py -> PPTX / DOCX / PDF / Quiz PDF (+answer key)
                  ▼
      Upload answer sheet ► Gemini reads it ► you correct it ► Grader agent/score
                  ▼
      < 50% ? ─► Training workflow on the weak topics ─► Remedial PPT/PDF/Word
```

## Step 1 - Install (Python 3.11)
```
python -m venv venv
venv\Scripts\activate          (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
CrewAI is large. If pip shows a conflict, run `pip install crewai==0.102.0` first, then the rest.

## Step 2 - Keys (never in code)
Create `.streamlit/secrets.toml`:
```
GROQ_API_KEY = "paste-here"
GEMINI_API_KEY = "paste-here"
TAVILY_API_KEY = "paste-here"
```
Create `.gitignore` with:
```
.streamlit/secrets.toml
venv/
__pycache__/
index/
```

## Step 3 - Run locally
```
streamlit run app.py
```
1. **📚 Knowledge Base** tab: either upload PDFs (pick manufacturer + model) or paste a Drive link, then press **Build knowledge base**.
2. Try **Troubleshooting**, **Training**, **Quiz**, **Score Answer Sheet**.

## Step 4 - GitHub and Streamlit
1. GitHub → New repository → upload `app.py`, `agents.py`, `documents.py`, `requirements.txt`, `README.md` (no manuals, no keys).
2. share.streamlit.io → New app → main file `app.py` → Advanced settings: Python 3.11 and paste the 3 keys in **Secrets**.
3. After deploy, open the app → 📚 tab → upload PDFs or give a Drive link.

## Limits to know (Streamlit Cloud is small, about 1 GB RAM)
- The knowledge base lives in memory and is lost when the app restarts or the browser session ends. Rebuild it from the 📚 tab.
- Start small: a few manuals, "Max pages per PDF" 300. Embedding runs on CPU (about 1-2 min per 1000 pages).
- Google Drive: gdown downloads about 50 files per folder and the whole folder before filtering. Use ONE sub-folder link. The folder must be shared "Anyone with the link can view".
- Scanned PDFs (no text) are skipped.

## Anti-hallucination
- Answers use only retrieved manual excerpts and cite Document | Page | Drive path.
- Weak match (`MIN_SCORE = 0.50` in `agents.py`) → "Not found in the OEM manuals" and a **Search online?** button. Tavily runs only after you click, and web results are shown separately.
- Images are illustrations from manual pages only. Values come from cited text.

## Common errors
| Problem | Fix |
|---|---|
| "Knowledge base is empty" | Build it in the 📚 tab first |
| Key missing message | Add the key to Secrets (cloud) or `.streamlit/secrets.toml` (local) |
| Everything says "Not found" | Lower `MIN_SCORE` to 0.40 in `agents.py`; check manufacturer/model; make sure the PDF has real text |
| Drive gives no PDFs | Folder not shared publicly, or link is wrong |
| Cloud app runs out of memory | Fewer PDFs / fewer pages per PDF |
| Deploy takes 10+ minutes | Normal for CrewAI the first time; check **Manage app → logs** |
| Handwriting misread | Fix it in the correction table before "Calculate final score" |
