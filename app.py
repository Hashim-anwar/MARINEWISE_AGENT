"""MarineWise AI - Streamlit UI only. Run locally with: streamlit run app.py"""
import pandas as pd
import streamlit as st

import agents as ag
import documents as docs

st.set_page_config(page_title="MarineWise AI", page_icon="⚓", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.5rem;}
.hero {background: linear-gradient(90deg,#0b2545,#13a89e); color: white; padding: 1.2rem 1.6rem;
       border-radius: 14px; margin-bottom: 1rem;}
.hero h1 {color: white; margin: 0;} .hero p {margin: 0; opacity: .9;}
div[data-testid="stTabs"] button {font-weight: 600;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>⚓ MarineWise AI</h1>'
            '<p>From OEM Knowledge to Skilled Technicians</p></div>', unsafe_allow_html=True)

MAKERS = ["CAT", "MTU", "YANMAR", "YAMAHA", "HONDA", "MAN", "OTHER"]
DRIVE_LINK = "https://drive.google.com/drive/folders/1gI_rJpsDzJ3HsOBO0Hb50B71vaJDkXY5"
S = st.session_state
ag.load_saved()  # reload an index saved earlier (if any)
for k in ("ts", "web", "train_files", "quiz", "quiz_pdf", "detected", "result", "remedial"):
    S.setdefault(k, None)


def badge(ok: bool, text: str) -> None:
    """Show a green/red status line in the sidebar."""
    (st.sidebar.success if ok else st.sidebar.error)(text)


st.sidebar.header("Status")
kb = ag.get_kb()
badge(kb is not None, f"Knowledge base: {len(kb['chunks'])} chunks" if kb else "Knowledge base empty")
badge(bool(ag.get_key("GROQ_API_KEY")), "Groq AI key")
badge(bool(ag.get_key("GEMINI_API_KEY")), "Gemini (vision) key")
badge(bool(ag.get_key("TAVILY_API_KEY")), "Tavily (web) key")
if kb is None:
    st.warning("Knowledge base is empty. Open the 📚 Knowledge Base tab first.")

tab0, tab1, tab2, tab3, tab4 = st.tabs(["📚 Knowledge Base", "🛠️ Troubleshooting", "🎓 Training",
                                        "📝 Quiz", "📷 Score Answer Sheet"])

# ------------------------------------------------------------ 0. Knowledge base
with tab0:
    st.write("Choose where your OEM manuals come from. They are read into memory (nothing is stored in GitHub).")
    src = st.radio("Source", ["📤 Upload PDFs here", "☁️ Google Drive link"], horizontal=True)
    max_pages = st.slider("Max pages read per PDF", 50, 1000, 300, 50)
    files, url, keyword, max_files = [], "", "", 10
    if src.startswith("📤"):
        u1, u2 = st.columns(2)
        up_maker = u1.selectbox("Manufacturer", MAKERS, key="up_maker")
        up_model = u2.text_input("Engine model (optional)", key="up_model")
        files = st.file_uploader("Manual PDFs", type=["pdf"], accept_multiple_files=True)
    else:
        url = st.text_input("Drive folder link (must be 'Anyone with the link can view')", DRIVE_LINK)
        keyword = st.text_input("Only PDFs whose path contains (optional)", placeholder="MTU")
        max_files = st.slider("Max PDFs to index", 1, 30, 10)
        st.caption("Tip: paste the link of ONE sub-folder (e.g. MTU) - faster and safer. "
                   "Google allows about 50 files per folder.")
    if st.button("⚙️ Build knowledge base", type="primary"):
        bar = st.progress(0.0)
        try:
            if src.startswith("📤"):
                entries = ag.entries_from_uploads(files, up_maker, up_model) if files else []
            else:
                with st.spinner("Downloading from Google Drive..."):
                    entries = ag.entries_from_drive(url, keyword, max_files)
            if not entries:
                st.warning("No PDFs found. Upload files or check the Drive link/sharing.")
            else:
                added = ag.build_kb(entries, max_pages, lambda f, t: bar.progress(min(float(f), 1.0), t))
                if added:
                    st.success(f"Added {added} chunks from {len(entries)} PDF(s).")
                    st.rerun()
                else:
                    st.warning("No readable text found (scanned PDFs are skipped).")
        except Exception as e:
            st.error(f"Could not build the knowledge base: {e}")
    kb = ag.get_kb()
    if kb:
        st.info(f"In memory: {len(kb['chunks'])} chunks from {len({c['doc'] for c in kb['chunks']})} document(s).")
        if st.button("🗑️ Clear knowledge base"):
            del S["kb"]
            st.rerun()

# ------------------------------------------------------------ 1. Troubleshooting
with tab1:
    c1, c2, c3 = st.columns(3)
    maker = c1.selectbox("Manufacturer", MAKERS)
    model = c2.text_input("Engine model", placeholder="e.g. 16V 4000 M90")
    serial = c3.text_input("Serial number (optional)")
    defect = st.text_area("Defect or alarm occurred", placeholder="e.g. Low fuel rail pressure alarm")
    if st.button("🔍 Search OEM manuals", type="primary"):
        S.web = None
        if not model or not defect:
            st.warning("Please enter the engine model and the defect/alarm.")
        else:
            with st.spinner("Searching manuals..."):
                try:
                    S.ts = ag.troubleshoot(maker, model, serial, defect)
                except Exception as e:
                    S.ts = None
                    st.error(f"Could not finish: {e}")
    r = S.ts
    if r and r["found"]:
        st.subheader("Answer from OEM manuals")
        st.markdown(r["answer"])
        with st.expander("📚 Sources (Document | Page | Drive path)"):
            for s in r["sources"]:
                st.write("• " + s)
    elif r:
        st.warning("Not found in the OEM manuals.")
        if st.button("🌐 Search online?"):
            with st.spinner("Searching the web..."):
                try:
                    S.web = ag.web_search(maker, model, defect)
                except Exception as e:
                    st.error(f"Web search failed: {e}")
    if S.web:
        st.subheader("🌐 Web result (NOT from OEM manual - verify before use)")
        st.markdown(S.web["answer"])
        with st.expander("Web sources"):
            for u in S.web["urls"]:
                st.write("• " + u)

# ------------------------------------------------------------ 2. Training
with tab2:
    c1, c2 = st.columns(2)
    t_model = c1.text_input("Engine model", key="t_model", placeholder="MTU 16V 4000 M90")
    ship = c2.text_input("Ship", key="ship")
    topic = st.text_input("Training topic", placeholder="Fuel Injection System")
    level = st.selectbox("Level", ["beginner", "intermediate"])
    st.write("Download as:")
    d1, d2, d3 = st.columns(3)
    want = {"PPT": d1.checkbox("PowerPoint (.pptx)", True), "Word": d2.checkbox("Word (.docx)"),
            "PDF": d3.checkbox("PDF", True)}
    if st.button("🎓 Generate training", type="primary"):
        if not t_model or not topic:
            st.warning("Please enter the engine model and topic.")
        else:
            with st.spinner("Reading manuals and building your training..."):
                try:
                    content = ag.make_training(t_model, ship, topic, level)
                    if content is None:
                        S.train_files = None
                        st.warning("Not enough information in the OEM manuals for this topic.")
                    else:
                        files = {}
                        if want["PPT"]:
                            files["Training.pptx"] = docs.build_pptx(content)
                        if want["Word"]:
                            files["Training.docx"] = docs.build_docx(content)
                        if want["PDF"]:
                            files["Training.pdf"] = docs.build_pdf(content)
                        S.train_files = files
                except Exception as e:
                    st.error(f"Could not finish: {e}")
    for name, data in (S.train_files or {}).items():
        st.download_button(f"⬇️ Download {name}", data, file_name=name, key="tr_" + name)

# ------------------------------------------------------------ 3. Quiz
with tab3:
    c1, c2 = st.columns(2)
    q_model = c1.text_input("Engine model", key="q_model", placeholder="MTU 16V 4000 M90")
    q_topic = c2.text_input("Quiz topic", placeholder="Fuel Injection System")
    c3, c4 = st.columns(2)
    q_type = c3.radio("Type", ["MCQ", "True/False", "Short answer"], horizontal=True)
    q_n = c4.slider("Number of questions", 5, 30, 10)
    if st.button("📝 Generate quiz PDF", type="primary"):
        if not q_model or not q_topic:
            st.warning("Please enter the engine model and topic.")
        else:
            with st.spinner("Writing questions from the manuals..."):
                try:
                    quiz = ag.make_quiz(q_model, q_topic, q_type, q_n)
                    if quiz is None:
                        st.warning("Not enough information in the OEM manuals for this topic.")
                    else:
                        S.quiz = {"items": quiz, "model": q_model, "topic": q_topic}
                        S.quiz_pdf = docs.build_quiz_pdf(quiz, f"{q_model} - {q_topic} Assessment")
                        S.detected, S.result, S.remedial = None, None, None
                except Exception as e:
                    st.error(f"Could not finish: {e}")
    if S.quiz_pdf:
        st.success(f"Quiz ready: {len(S.quiz['items'])} questions.")
        st.download_button("⬇️ Download Quiz PDF (with answer key)", S.quiz_pdf,
                           file_name="Quiz.pdf", key="dl_quiz")

# ------------------------------------------------------------ 4. Score answer sheet
with tab4:
    if not S.quiz:
        st.info("Generate a quiz first (Quiz tab). The score is checked against its answer key.")
    else:
        up = st.file_uploader("Upload photo of the completed ANSWER SHEET", type=["jpg", "jpeg", "png"])
        if up and st.button("🔎 Read answers", type="primary"):
            with st.spinner("Reading handwriting..."):
                try:
                    S.detected = ag.read_answer_sheet(up.getvalue(), up.type, len(S.quiz["items"]))
                    S.result, S.remedial = None, None
                except Exception as e:
                    st.error(f"Could not read the image: {e}")
        if S.detected is not None:
            st.write("✏️ Check and correct any answer the AI misread, then calculate the score:")
            n = len(S.quiz["items"])
            df = pd.DataFrame({"Q": range(1, n + 1),
                               "Answer": [str(S.detected.get(str(i), "")) for i in range(1, n + 1)]})
            edited = st.data_editor(df, hide_index=True, disabled=["Q"], key="editor")
            if st.button("✅ Calculate final score"):
                given = {str(q): a for q, a in zip(edited["Q"], edited["Answer"])}
                with st.spinner("Scoring..."):
                    try:
                        S.result = ag.grade(S.quiz["items"], given)
                        S.remedial = None
                        if S.result["percent"] < 50:
                            weak = "; ".join(S.result["wrong"][:6])
                            content = ag.make_training(S.quiz["model"], "",
                                                       f"{S.quiz['topic']} - weak areas: {weak}", "beginner")
                            if content:
                                S.remedial = {"Remedial_Training.pptx": docs.build_pptx(content),
                                              "Remedial_Training.pdf": docs.build_pdf(content),
                                              "Remedial_Training.docx": docs.build_docx(content)}
                    except Exception as e:
                        st.error(f"Scoring failed: {e}")
        res = S.result
        if res:
            st.metric("Score", f"{res['percent']}%", f"{res['correct']} / {res['total']} correct")
            if res["percent"] >= 50:
                st.success("PASS ✅")
            else:
                st.error("Below 50% - remedial training prepared")
                for name, data in (S.remedial or {}).items():
                    st.download_button(f"⬇️ Download {name}", data, file_name=name, key="rm_" + name)
            if res["wrong"]:
                with st.expander("Questions answered wrongly"):
                    for w in res["wrong"]:
                        st.write("• " + w)
