# ⚓ MarineWise AI (simple MVP)

Troubleshooting from OEM manuals → Training (PPT/Word/PDF) → Quiz PDF + answer key → Answer-sheet scoring → Remedial training.

## Your files (only 6)

| File | What it does |
|---|---|
| `app.py` | The screen (4 tabs). Only UI. |
| `agents.py` | Keys, FAISS + BM25 search, and the **CrewAI** agents (each job = a Crew of 2 agents: a writer + a Quality Reviewer) |
| `documents.py` | Makes PPTX, DOCX, PDF and Quiz PDF |
| `build_index.py` | Run ONCE on your PC: downloads manuals and builds the search index |
| `requirements.txt` | Libraries |
| `README.md` | This guide |

## IMPORTANT: why the index is built on your PC
Streamlit Cloud is small (about 1 GB RAM). It cannot download or read 2 GB of manuals.
So: **build the index on your computer → upload only the small `index/` folder → the cloud app only searches it.**
(The scripts were written carefully but were NOT run by me, as I had no internet. If a pinned version fails to install, remove its `==version` and retry.)

## CrewAI note
CrewAI is heavy. Use Python 3.11 (3.10-3.12 work). If `pip install` complains about a conflict, install `crewai` first, then the rest. Groq is used through CrewAI (`groq/openai/gpt-oss-120b`), so there is no separate `groq` package.

## Step 1 - Start fresh
Delete your old messy project. Make a new folder `marinewise` and put the 5 code files in it.

## Step 2 - Install (Python 3.11 recommended)
```
cd marinewise
python -m venv venv
venv\Scripts\activate          (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Step 3 - Add your keys (never inside code)
Create folder `.streamlit` and a file `.streamlit/secrets.toml`:
```
GROQ_API_KEY = "paste-here"
GEMINI_API_KEY = "paste-here"
TAVILY_API_KEY = "paste-here"
```
Create a file `.gitignore` containing:
```
.streamlit/secrets.toml
manuals/
venv/
__pycache__/
```
Never upload `secrets.toml` or `manuals/` to GitHub.

## Step 4 - Build the index (on your PC)
Test small first (one manufacturer):
```
python build_index.py MTU
```
When it works, run everything: `python build_index.py`
- gdown downloads only ~50 files per folder. If some are missing, download them by hand from Drive and put the PDFs in `manuals/` (any sub-folders), then run the script again.
- Scanned PDFs (no text) are skipped and logged.
- Check size: `du -sh index` (Mac/Linux) or right-click the `index` folder → Properties (Windows).

## Step 5 - Run locally
```
streamlit run app.py
```
The sidebar should show green badges. Try: Manufacturer MTU, a model, defect "low fuel pressure".

## Step 6 - Put it on GitHub
1. github.com → New repository (Public) → `marinewise`.
2. "Add file → Upload files": drag `app.py`, `agents.py`, `documents.py`, `requirements.txt`, `README.md` and the whole `index/` folder.
3. **GitHub web upload limit is 25 MB per file.** If `faiss.index` or `chunks.json` is bigger, either index fewer/most important manuals (`python build_index.py MTU`), or use Git LFS (`git lfs install`, `git lfs track "index/*"`, then `git add`, `git commit`, `git push`).

## Step 7 - Deploy
1. share.streamlit.io → New app → choose your repo, branch `main`, main file `app.py`.
2. Advanced settings → Python 3.11 → paste your 3 keys in **Secrets** (same format as Step 3).
3. Deploy. First start downloads a small embedding model (1-2 minutes).

## How it avoids hallucination
- Answers use only retrieved manual chunks and cite Document | Page | Drive path.
- If the best match is weak (`MIN_SCORE = 0.50` in `agents.py`) → "Not found in the OEM manuals" + **Search online?** button. Tavily runs only after you click.
- Web results are shown separately and labelled "NOT from OEM manual".
- Images are only illustrations pulled from manual pages. Values come from cited text.

## Common errors
| Problem | Fix |
|---|---|
| "No index found" | Upload the `index/` folder next to `app.py` |
| Key missing message | Add the key to Secrets (cloud) or `.streamlit/secrets.toml` (local) |
| Everything says "Not found" | Lower `MIN_SCORE` to 0.40 in `agents.py`, or check the manufacturer/model spelling |
| Cloud app crashes out of memory | Index fewer manuals |
| Handwriting misread | Fix it in the correction table before pressing "Calculate final score" |
