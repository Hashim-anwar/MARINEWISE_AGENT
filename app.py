import hashlib
import io
import re
import string
from pathlib import Path

import faiss
import numpy as np
import requests
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI OEM Manuals Assistant",
    page_icon="⚓",
    layout="wide",
)


# =========================================================
# CONSTANTS
# =========================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

DEFAULT_TOP_K = 6
SEMANTIC_CANDIDATES = 50

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but",
    "by", "can", "could", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "i", "if", "in", "into", "is",
    "it", "its", "may", "me", "my", "of", "on", "or", "our",
    "please", "should", "that", "the", "their", "there", "this",
    "to", "was", "we", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your"
}


# =========================================================
# DOCUMENT EXTRACTION
# =========================================================

def extract_pdf(file):
    """Extract PDF text page by page."""
    results = []
    reader = PdfReader(file)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            results.append({
                "text": text,
                "filename": file.name,
                "page": page_number,
            })

    return results


def extract_docx(file):
    """Extract text from a DOCX file."""
    document = Document(io.BytesIO(file.read()))

    paragraphs = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    text = "\n".join(paragraphs)

    return [{
        "text": text,
        "filename": file.name,
        "page": None,
    }] if text.strip() else []


def extract_txt(file):
    """Extract text from a TXT file."""
    text = file.read().decode("utf-8", errors="ignore")

    return [{
        "text": text,
        "filename": file.name,
        "page": None,
    }] if text.strip() else []


def extract_md(file):
    """Extract text from a Markdown file."""
    text = file.read().decode("utf-8", errors="ignore")

    return [{
        "text": text,
        "filename": file.name,
        "page": None,
    }] if text.strip() else []


def extract_document(file):
    """Route a document to its correct extractor."""
    extension = Path(file.name).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file)

    if extension == ".docx":
        return extract_docx(file)

    if extension == ".txt":
        return extract_txt(file)

    if extension == ".md":
        return extract_md(file)

    return []


# =========================================================
# CHUNKING
# =========================================================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character-based chunks."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

        start += step

    return chunks


def create_chunks(extracted_documents):
    """Chunk extracted text while preserving document metadata."""
    chunks = []

    for document in extracted_documents:
        text_chunks = chunk_text(document["text"])

        for chunk in text_chunks:
            chunks.append({
                "filename": document["filename"],
                "page": document["page"],
                "text": chunk,
            })

    return chunks


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    """Load the embedding model once per Streamlit server."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts):
    """Create normalized embeddings for a list of texts."""
    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    return np.asarray(embeddings, dtype="float32")


# =========================================================
# FAISS
# =========================================================

def build_faiss_index(embeddings):
    """Build a cosine-similarity FAISS index."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def add_embeddings_to_index(embeddings):
    """Add new document embeddings to the existing FAISS index."""
    if st.session_state.faiss_index is None:
        st.session_state.faiss_index = build_faiss_index(embeddings)
    else:
        st.session_state.faiss_index.add(embeddings)


# =========================================================
# KEYWORD SEARCH
# =========================================================

def important_words(text):
    """Extract simple lowercase keywords from a question."""
    cleaned = text.lower().translate(
        str.maketrans("", "", string.punctuation)
    )

    return [
        word
        for word in cleaned.split()
        if word not in STOPWORDS and len(word) > 1
    ]


def keyword_score(query, chunk_text_value):
    """Calculate a simple keyword match score."""
    query_words = important_words(query)

    if not query_words:
        return 0.0

    chunk_words = re.findall(r"\b\w+\b", chunk_text_value.lower())

    if not chunk_words:
        return 0.0

    chunk_word_counts = {}

    for word in chunk_words:
        chunk_word_counts[word] = chunk_word_counts.get(word, 0) + 1

    score = 0.0

    for word in query_words:
        score += min(chunk_word_counts.get(word, 0), 3)

    return score / len(query_words)


# =========================================================
# HYBRID SEARCH
# =========================================================

def normalize_scores(values):
    """Normalize scores to a 0-1 range."""
    if not values:
        return {}

    minimum = min(values.values())
    maximum = max(values.values())

    if maximum == minimum:
        if maximum > 0:
            return {key: 1.0 for key in values}
        return {key: 0.0 for key in values}

    return {
        key: (value - minimum) / (maximum - minimum)
        for key, value in values.items()
    }


def hybrid_search(query, top_k=DEFAULT_TOP_K):
    """Combine semantic and keyword search into one ranking."""
    if not st.session_state.chunks:
        return []

    query_embedding = embed_texts([query])

    semantic_count = min(
        SEMANTIC_CANDIDATES,
        len(st.session_state.chunks)
    )

    distances, indices = st.session_state.faiss_index.search(
        query_embedding,
        semantic_count,
    )

    semantic_scores = {}

    for score, index_number in zip(distances[0], indices[0]):
        if index_number >= 0:
            semantic_scores[int(index_number)] = float(score)

    keyword_scores = {
        index_number: keyword_score(query, chunk["text"])
        for index_number, chunk in enumerate(st.session_state.chunks)
    }

    # Include the strongest keyword matches even if semantic search
    # did not place them in its candidate set.
    strongest_keyword_indices = sorted(
        keyword_scores,
        key=keyword_scores.get,
        reverse=True,
    )[:SEMANTIC_CANDIDATES]

    candidate_indices = set(semantic_scores)
    candidate_indices.update(strongest_keyword_indices)

    candidate_semantic = {
        index_number: semantic_scores.get(index_number, 0.0)
        for index_number in candidate_indices
    }

    candidate_keyword = {
        index_number: keyword_scores.get(index_number, 0.0)
        for index_number in candidate_indices
    }

    semantic_normalized = normalize_scores(candidate_semantic)
    keyword_normalized = normalize_scores(candidate_keyword)

    ranked = []

    for index_number in candidate_indices:
        semantic = semantic_normalized.get(index_number, 0.0)
        keyword = keyword_normalized.get(index_number, 0.0)

        combined = (0.7 * semantic) + (0.3 * keyword)

        chunk = st.session_state.chunks[index_number].copy()
        chunk["score"] = combined
        chunk["semantic_score"] = semantic
        chunk["keyword_score"] = keyword

        ranked.append(chunk)

    ranked.sort(key=lambda item: item["score"], reverse=True)

    return ranked[:top_k]


# =========================================================
# GROQ
# =========================================================

def get_groq_client():
    """Create a Groq client from Streamlit secrets."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        return None

    if not api_key:
        return None

    return Groq(api_key=api_key)


def format_source_label(chunk):
    """Create a readable source label."""
    if chunk["page"] is not None:
        return f'{chunk["filename"]}, Page {chunk["page"]}'

    return chunk["filename"]


def answer_question(question, retrieved_chunks):
    """Ask Groq to answer strictly from retrieved manual context."""
    client = get_groq_client()

    if client is None:
        return (
            "Groq API key is not configured. Add GROQ_API_KEY "
            "to Streamlit secrets."
        )

    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[SOURCE {number}: {format_source_label(chunk)}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are an AI OEM Manuals Assistant.

Answer the user's question ONLY using the provided manual context.

Strict rules:
1. Do not use outside knowledge.
2. Do not guess, invent, or fill missing information.
3. If the answer is not supported by the provided context, clearly say:
   "The provided manuals do not contain enough information to answer this."
4. If the context contains conflicting information, state the conflict.
5. Preserve technical values, part numbers, alarm codes, units, limits,
   procedures, and warnings exactly when they are supported by the context.
6. Give a clear, practical answer suitable for a marine engineer or technician.
7. When useful, mention the source filename and page shown in the context.
"""

    user_prompt = f"""QUESTION:
{question}

MANUAL CONTEXT:
{context}
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    except Exception as error:
        return f"Groq request failed: {error}"


# =========================================================
# GOOGLE DRIVE
# =========================================================

def extract_drive_id(link):
    """Extract a Google Drive file or folder ID from common URLs."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/folders/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, link)

        if match:
            return match.group(1)

    return None


def get_drive_api_key():
    """Read the optional Google Drive API key from Streamlit secrets."""
    try:
        return st.secrets["GOOGLE_DRIVE_API_KEY"]
    except Exception:
        return None


def drive_request(url, params=None):
    """Make a Google Drive REST request with the configured API key."""
    api_key = get_drive_api_key()

    if not api_key:
        raise RuntimeError(
            "GOOGLE_DRIVE_API_KEY is not configured in Streamlit secrets."
        )

    response = requests.get(
        url,
        params={**(params or {}), "key": api_key},
        timeout=60,
    )

    response.raise_for_status()
    return response


def drive_metadata(file_id):
    """Get metadata for a Google Drive item."""
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"

    response = drive_request(
        url,
        {
            "fields": (
                "id,name,mimeType,size,trashed,"
                "webViewLink,parents"
            )
        },
    )

    return response.json()


def list_drive_children(folder_id):
    """List supported files and folders directly inside a Drive folder."""
    url = "https://www.googleapis.com/drive/v3/files"

    files = []
    page_token = None

    while True:
        params = {
            "q": (
                f"'{folder_id}' in parents "
                "and trashed = false"
            ),
            "fields": (
                "nextPageToken,"
                "files(id,name,mimeType,size,trashed)"
            ),
            "pageSize": 1000,
        }

        if page_token:
            params["pageToken"] = page_token

        response = drive_request(url, params)
        data = response.json()

        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")

        if not page_token:
            break

    return files


def collect_drive_files(folder_id):
    """Recursively collect supported files from a Drive folder."""
    supported_extensions = {".pdf", ".docx", ".txt", ".md"}
    collected = []

    for item in list_drive_children(folder_id):
        name = item.get("name", "")
        extension = Path(name).suffix.lower()

        if item["mimeType"] == "application/vnd.google-apps.folder":
            collected.extend(collect_drive_files(item["id"]))

        elif extension in supported_extensions:
            collected.append(item)

    return collected


def download_drive_file(file_id, filename):
    """Download a supported Google Drive file."""
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"

    response = drive_request(
        url,
        {"alt": "media"},
    )

    file_object = io.BytesIO(response.content)
    file_object.name = filename

    return file_object


# =========================================================
# DOCUMENT PROCESSING
# =========================================================

def file_hash(file_bytes):
    """Create a stable hash to prevent duplicate processing."""
    return hashlib.sha256(file_bytes).hexdigest()


def process_new_file(file_object, source_name=None):
    """Extract, chunk, embed, and index one new file."""
    raw_bytes = file_object.getvalue()

    document_hash = file_hash(raw_bytes)

    if document_hash in st.session_state.processed_hashes:
        return 0, False

    extraction_file = io.BytesIO(raw_bytes)
    extraction_file.name = source_name or file_object.name

    extracted = extract_document(extraction_file)

    chunks = create_chunks(extracted)

    if not chunks:
        return 0, False

    embeddings = embed_texts([chunk["text"] for chunk in chunks])

    add_embeddings_to_index(embeddings)

    st.session_state.chunks.extend(chunks)
    st.session_state.processed_hashes.add(document_hash)

    return len(chunks), True


def process_uploaded_files(uploaded_files):
    """Process all newly uploaded local files."""
    total_new_chunks = 0
    new_documents = 0

    for uploaded_file in uploaded_files:
        try:
            raw_bytes = uploaded_file.getvalue()

            file_object = io.BytesIO(raw_bytes)
            file_object.name = uploaded_file.name

            chunks_created, was_new = process_new_file(file_object)

            if was_new:
                total_new_chunks += chunks_created
                new_documents += 1

        except Exception as error:
            st.error(
                f"Could not process {uploaded_file.name}: {error}"
            )

    return new_documents, total_new_chunks


def process_drive_link(link):
    """Load a Drive file or folder and process its supported documents."""
    drive_id = extract_drive_id(link.strip())

    if not drive_id:
        raise ValueError(
            "Could not find a Google Drive file or folder ID in the link."
        )

    metadata = drive_metadata(drive_id)

    if metadata.get("trashed"):
        raise ValueError("The Google Drive item is in the trash.")

    if metadata["mimeType"] == "application/vnd.google-apps.folder":
        drive_files = collect_drive_files(drive_id)
    else:
        extension = Path(metadata["name"]).suffix.lower()

        if extension not in {".pdf", ".docx", ".txt", ".md"}:
            raise ValueError(
                "This Drive item is not a supported PDF, DOCX, TXT, or MD file."
            )

        drive_files = [metadata]

    if not drive_files:
        return 0, 0

    new_documents = 0
    total_new_chunks = 0

    for item in drive_files:
        try:
            file_object = download_drive_file(
                item["id"],
                item["name"],
            )

            chunks_created, was_new = process_new_file(
                file_object,
                source_name=item["name"],
            )

            if was_new:
                new_documents += 1
                total_new_chunks += chunks_created

        except Exception as error:
            st.error(
                f"Could not load {item['name']} from Google Drive: {error}"
            )

    return new_documents, total_new_chunks


# =========================================================
# SESSION STATE
# =========================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = set()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "drive_link" not in st.session_state:
    st.session_state.drive_link = ""


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("📚 OEM Manuals")

    uploaded_files = st.file_uploader(
        "Upload manuals",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, TXT and Markdown.",
    )

    if uploaded_files:
        if st.button("Process Uploaded Manuals", use_container_width=True):
            with st.spinner("Extracting, chunking, embedding and indexing..."):
                documents, chunks = process_uploaded_files(uploaded_files)

            if documents:
                st.success(
                    f"Added {documents} new document(s) and {chunks} chunks."
                )
            else:
                st.info("No new documents were added.")

    st.divider()

    st.header("☁️ Google Drive")

    drive_link = st.text_input(
        "Public/shared Drive file or folder link",
        value=st.session_state.drive_link,
        placeholder="Paste Google Drive link here",
    )

    if st.button("Load from Google Drive", use_container_width=True):
        if not drive_link.strip():
            st.warning("Paste a Google Drive file or folder link first.")
        else:
            with st.spinner(
                "Loading Drive files, extracting, chunking and indexing..."
            ):
                try:
                    documents, chunks = process_drive_link(drive_link)

                    if documents:
                        st.success(
                            f"Added {documents} Drive document(s) and "
                            f"{chunks} chunks."
                        )
                    else:
                        st.info("No new Drive documents were added.")

                except Exception as error:
                    st.error(f"Google Drive error: {error}")

    st.divider()

    st.header("⚙️ Search")

    top_k = st.slider(
        "Sources per answer",
        min_value=3,
        max_value=10,
        value=DEFAULT_TOP_K,
    )

    if st.session_state.chunks:
        st.metric("Indexed Chunks", len(st.session_state.chunks))

    if st.button("Clear Knowledge Base", use_container_width=True):
        st.session_state.chunks = []
        st.session_state.faiss_index = None
        st.session_state.processed_hashes = set()
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.caption(
        "Answers are grounded only in the indexed OEM manuals."
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title("⚓ AI OEM Manuals Assistant")

st.write(
    "Ask questions about your OEM manuals and receive "
    "evidence-backed answers using hybrid semantic and keyword search."
)


# =========================================================
# STATUS
# =========================================================

if not st.session_state.chunks:
    st.info(
        "No manuals are indexed yet. Upload a manual from the sidebar "
        "or load a supported public/shared Google Drive file or folder."
    )
else:
    st.success(
        f"Knowledge base ready: "
        f"{len(st.session_state.chunks)} chunks indexed."
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources"):
                for number, source in enumerate(
                    message["sources"],
                    start=1,
                ):
                    st.markdown(
                        f"**Source {number}: "
                        f"{format_source_label(source)}**"
                    )
                    st.caption(
                        f"Hybrid score: {source['score']:.3f}"
                    )
                    st.text(
                        source["text"][:1200]
                        + (
                            "..."
                            if len(source["text"]) > 1200
                            else ""
                        )
                    )


# =========================================================
# QUESTION INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about the uploaded OEM manuals..."
)

if question:
    if not st.session_state.chunks:
        st.warning(
            "Please upload or load at least one OEM manual before asking a question."
        )
    else:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the OEM manuals..."):
                retrieved_chunks = hybrid_search(
                    question,
                    top_k=top_k,
                )

            if not retrieved_chunks:
                answer = (
                    "The provided manuals do not contain enough information "
                    "to answer this."
                )
            else:
                with st.spinner("Generating answer from the retrieved evidence..."):
                    answer = answer_question(
                        question,
                        retrieved_chunks,
                    )

            st.markdown(answer)

            with st.expander("📚 Sources", expanded=True):
                for number, source in enumerate(
                    retrieved_chunks,
                    start=1,
                ):
                    st.markdown(
                        f"**Source {number}: "
                        f"{format_source_label(source)}**"
                    )
                    st.caption(
                        f"Hybrid score: {source['score']:.3f}"
                    )
                    st.text(
                        source["text"][:1200]
                        + (
                            "..."
                            if len(source["text"]) > 1200
                            else ""
                        )
                    )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved_chunks,
        })
