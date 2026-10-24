import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi

from src.answer_question import (
    EMBEDDINGS_FILE,
    CHUNKS_FILE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    load_chunks,
    tokenize,
    retrieve_chunks,
    generate_answer,
)

# =========================================================
# 1. CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHAT_DIR = DATA_DIR / "chat_sessions"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHAT_DIR.mkdir(parents=True, exist_ok=True)

# Initial settings to evaluate. There is no universally best chunk size;
# compare retrieval metrics on your evaluation questions before finalizing.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
MAX_SOURCES_IN_PROMPT = 6
SEARCH_MODES = ["Uploaded PDFs", "Built-in papers", "Both"]

# =========================================================
# 2. CHAT HISTORY STORAGE
# =========================================================
def save_chat(chat):
    """Persist one conversation to its own JSON file."""
    chat["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path = CHAT_DIR / f"{chat['id']}.json"
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(chat, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def load_all_chats():
    """Load saved conversations without deleting any."""
    chats = {}
    for path in CHAT_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                chat = json.load(file)
            if "id" not in chat or "messages" not in chat:
                continue
            # Add defaults so chats saved by an earlier app version still work.
            chat.setdefault("title", "New Chat")
            chat.setdefault("created_at", chat.get("updated_at", "Unknown"))
            chat.setdefault("updated_at", chat.get("created_at", ""))
            chat.setdefault("search_mode", "Uploaded PDFs")
            chat.setdefault("pending_question", None)
            chat.setdefault("pending_status", None)
            chat.setdefault("last_error", None)
            chats[chat["id"]] = chat
        except (OSError, json.JSONDecodeError):
            continue
    return chats


def create_chat():
    """Create a new, independent conversation."""
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "id": uuid.uuid4().hex,
        "title": "New Chat",
        "created_at": now,
        "updated_at": now,
        "search_mode": "Uploaded PDFs",
        "messages": [],
        "pending_question": None,
        "pending_status": None,
        "last_error": None,
    }


def make_chat_title(question):
    """Create a short title without calling an LLM."""
    title = re.sub(r"\s+", " ", question).strip().rstrip(" ?!.")
    if not title:
        return "New Chat"
    title = re.sub(
        r"^(can you explain|could you explain|please explain|can you tell me|"
        r"could you tell me|tell me about|explain to me|what is|what are|"
        r"how does|how do|how can|why does|why do)\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = title.strip().capitalize()
    if len(title) > 55:
        title = title[:52].rsplit(" ", 1)[0] + "..."
    return title or "New Chat"


# =========================================================
# 3. INITIALIZE SESSION STATE
# =========================================================
if "chats" not in st.session_state:
    st.session_state.chats = load_all_chats()
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "upload_index" not in st.session_state:
    st.session_state.upload_index = None
if "upload_fingerprint" not in st.session_state:
    st.session_state.upload_fingerprint = None
if "upload_error" not in st.session_state:
    st.session_state.upload_error = None


def get_active_chat():
    """Return the currently selected chat."""
    chat_id = st.session_state.active_chat_id
    if chat_id is None:
        return None
    return st.session_state.chats.get(chat_id)


def persist_active_chat():
    """Save the current chat if one exists."""
    chat = get_active_chat()
    if chat is not None:
        save_chat(chat)


def start_new_chat():
    """Create a new chat without modifying previous chats."""
    chat = create_chat()
    st.session_state.chats[chat["id"]] = chat
    st.session_state.active_chat_id = chat["id"]
    save_chat(chat)


# =========================================================
# 4. LOAD BUILT-IN RAG SYSTEM
# =========================================================
@st.cache_resource(show_spinner="Loading retrieval models...")
def load_rag_system():
    """Load built-in chunks, embeddings, BM25 and encoder once per app process."""

    # Import the heavy model library only when retrieval is needed.
    from sentence_transformers import SentenceTransformer

    embeddings = np.load(EMBEDDINGS_FILE)
    chunks = load_chunks(CHUNKS_FILE)

    if len(embeddings) != len(chunks):
        raise ValueError(
            "The built-in embeddings and chunks do not match. "
            "Rebuild the embeddings using your project pipeline."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "The built-in embeddings must be a 2D array."
        )

    bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return embeddings, chunks, bm25, embedding_model


# =========================================================
# 5. PERSIST UPLOADED PDF FILES
# =========================================================
def persist_uploaded_files(uploaded_files):
    """Store uploaded PDFs in data/uploads/ so they remain after a restart."""
    for uploaded_file in uploaded_files or []:
        filename = Path(uploaded_file.name).name
        if not filename.lower().endswith(".pdf"):
            continue
        pdf_bytes = uploaded_file.getvalue()
        if not pdf_bytes:
            continue
        destination = UPLOAD_DIR / filename
        if destination.exists() and destination.read_bytes() == pdf_bytes:
            continue
        destination.write_bytes(pdf_bytes)


def get_saved_pdfs():
    """Return saved PDFs as (filename, bytes) pairs."""
    pdf_records = []
    for path in sorted(UPLOAD_DIR.glob("*.pdf")):
        try:
            pdf_records.append((path.name, path.read_bytes()))
        except OSError:
            continue
    return pdf_records


def get_pdf_fingerprint(pdf_records):
    """Identify changes in the saved document collection."""
    digest = hashlib.sha256()
    for filename, pdf_bytes in pdf_records:
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(pdf_bytes).digest())
    return digest.hexdigest()


# =========================================================
# 6. EXTRACT AND CHUNK PDF TEXT
# =========================================================
def process_uploaded_pdfs(pdf_records, embedding_model):
    """Extract text from each PDF and create overlapping, page-aware chunks."""
    uploaded_chunks = []
    chunk_number = 0

    for filename, pdf_bytes in pdf_records:
        try:
            document = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as error:
            raise ValueError(f"Could not open '{filename}': {error}") from error

        with document:
            for page_number, page in enumerate(document, start=1):
                text = re.sub(r"\s+", " ", page.get_text("text")).strip()
                if not text:
                    continue

                start = 0
                chunk_index = 0
                while start < len(text):
                    end = min(start + CHUNK_SIZE, len(text))

                    # Prefer ending near a sentence/word boundary when possible.
                    if end < len(text):
                        lower_bound = start + int(CHUNK_SIZE * 0.65)
                        boundary_candidates = [
                            text.rfind(mark, lower_bound, end)
                            for mark in (". ", "? ", "! ", "; ", ", ", " ")
                        ]
                        best_boundary = max(boundary_candidates)
                        if best_boundary > start:
                            end = best_boundary + 1

                    passage = text[start:end].strip()
                    if passage:
                        uploaded_chunks.append({
                            "chunk_id": f"upload_{chunk_number}",
                            "chunk_index": chunk_index,
                            "source": filename,
                            "page": page_number,
                            "text": passage,
                        })
                        chunk_number += 1
                        chunk_index += 1

                    if end >= len(text):
                        break
                    next_start = max(0, end - CHUNK_OVERLAP)
                    # Safety guard: always move forward, even on unusual text.
                    start = next_start if next_start > start else end

    if not uploaded_chunks:
        raise ValueError(
            "No selectable text was found in the saved PDFs. "
            "They may be scanned documents that require OCR."
        )

    texts = [chunk["text"] for chunk in uploaded_chunks]
    uploaded_embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )
    uploaded_bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in uploaded_chunks])
    return uploaded_chunks, uploaded_embeddings, uploaded_bm25


# =========================================================
# 7. BUILD OR REUSE THE UPLOADED DOCUMENT INDEX
# =========================================================
def get_uploaded_index(pdf_records, embedding_model):
    """Reuse the index while the saved PDF collection is unchanged."""
    fingerprint = get_pdf_fingerprint(pdf_records)
    current_index = st.session_state.upload_index

    if current_index is not None and st.session_state.upload_fingerprint == fingerprint:
        return current_index
    if not pdf_records:
        st.session_state.upload_index = None
        st.session_state.upload_fingerprint = fingerprint
        return None

    with st.spinner("Indexing uploaded PDFs (only needed when documents change)..."):
        chunks, embeddings, bm25 = process_uploaded_pdfs(pdf_records, embedding_model)

    index = {"chunks": chunks, "embeddings": embeddings, "bm25": bm25}
    st.session_state.upload_index = index
    st.session_state.upload_fingerprint = fingerprint
    return index


# =========================================================
# 8. REMOVE DUPLICATE RETRIEVED PASSAGES
# =========================================================
def deduplicate_results(results):
    """Remove repeated passages without changing ranking order."""
    unique_results = []
    seen = set()
    for item in results:
        normalized_text = re.sub(r"\s+", " ", item.get("text", "")).strip().casefold()
        key = (item.get("source"), item.get("page"), normalized_text)
        if key in seen:
            continue
        seen.add(key)
        unique_results.append(item)
    return unique_results


# =========================================================
# 9. BUILD THE ANSWER PROMPT
# =========================================================

def build_app_prompt(question, results):
    """Build a grounded prompt with original source metadata."""

    evidence_parts = []

    for i, item in enumerate(
        results[:MAX_SOURCES_IN_PROMPT], start=1
    ):
        evidence_parts.append(
            f"[Evidence {i}]\n"
            f"Source: {item.get('source', 'Unknown')}\n"
            f"Page: {item.get('page', 'Unknown')}\n"
            f"Passage: {item.get('text', '')}"
        )

    evidence_text = "\n\n".join(evidence_parts)

    instructions = """
You are a research-paper question-answering assistant.

Answer the user's question directly using only the supplied evidence.

Rules:
1. Identify the exact fact that answers the question.
2. Support every factual claim with the supplied evidence.
3. Do not confuse alternative experiments or ablation studies
   with the original method proposed in the paper.
4. Never contradict information stated elsewhere in your answer.
5. If the evidence is insufficient, clearly state what is missing.
   Do not guess or add unsupported speculation.
6. Keep the answer concise but include the important details.
7. Do not repeat the same explanation in different words.
8. Include relevant source names and page numbers when available.
9. If different passages appear inconsistent, explain the
   uncertainty instead of silently choosing an unsupported answer.
10. Keep answers to approximately 80–100 words unless the question
requires more detail. Answer the question directly and finish
the explanation before adding optional background information.Try answering within a few seconds.
11.Use only the supplied evidence to answer the question.
Do not invent facts, citations, page numbers, or quotations.
If the evidence does not establish an answer, say so clearly.
Distinguish the paper's original method from comparison experiments.
12.Do not contradict yourself.
13.Cite the relevant source and page when available.
14.Before saying that information is absent from the evidence,
check whether the evidence already states or directly describes it.
Never claim that information is missing if you have just provided it.
15.Start with the direct answer. Do not begin by saying
"According to the supplied evidence" unless needed for clarity.
Avoid unnecessary technical claims that are not needed to answer
the question. Never add a conclusion that conflicts with the answer.
"""

    prompt = f"""
{instructions}

USER QUESTION:
{question}

SUPPLIED EVIDENCE:
{evidence_text}

Write the answer now. Follow the instructions and do not introduce
facts that are not supported by the supplied evidence.
"""

    return prompt.strip()


# =========================================================
# 10. RETRIEVE FROM THE SELECTED DOCUMENT COLLECTION
# =========================================================
def is_metadata_question(question):
    q = question.casefold()
    terms = (
        "author", "authors", "who wrote", "written by", "paper title",
        "title of the paper", "title of this paper", "publication year",
        "published in", "publication date", "abstract", "affiliation",
    )
    return any(term in q for term in terms)


def prioritize_paper_metadata(question, results, chunks):
    """For author/title/abstract questions, add relevant first-page passages first."""
    if not is_metadata_question(question):
        return results

    question_words = set(re.findall(r"[a-z0-9]+", question.casefold()))
    source_names = []
    for item in results:
        source = item.get("source")
        if source and source not in source_names:
            source_names.append(source)

    # If a filename is mentioned in the question, prioritize that document.
    filename_matches = []
    for chunk in chunks:
        source = chunk.get("source", "")
        source_words = set(re.findall(r"[a-z0-9]+", Path(source).stem.casefold()))
        meaningful = {word for word in source_words if len(word) > 2}
        if meaningful and len(meaningful & question_words) >= min(2, len(meaningful)):
            if source not in filename_matches:
                filename_matches.append(source)

    preferred_sources = filename_matches + [s for s in source_names if s not in filename_matches]
    if not preferred_sources:
        return results

    metadata_results = []
    seen = set()
    for source in preferred_sources:
        first_page = [
            chunk for chunk in chunks
            if chunk.get("source") == source and chunk.get("page") == 1
        ]
        for chunk in first_page:
            key = (chunk.get("source"), chunk.get("page"), chunk.get("text", ""))
            if key in seen:
                continue
            seen.add(key)
            metadata_results.append({
                **chunk,
                "dense_rank": "—",
                "bm25_rank": "—",
                "rrf_score": 0.0,
            })
        # Avoid pulling first pages from an excessive number of unrelated files.
        if len(metadata_results) >= 3:
            break

    return deduplicate_results(metadata_results + results)


def search_documents(question, search_mode, pdf_records):
    """Search uploaded papers, built-in papers, or both using dense + BM25 + RRF."""
    base_embeddings, base_chunks, base_bm25, embedding_model = load_rag_system()
    upload_index = None

    if search_mode in ("Uploaded PDFs", "Both"):
        upload_index = get_uploaded_index(pdf_records, embedding_model)

    if search_mode == "Uploaded PDFs":
        if upload_index is None:
            raise ValueError("No saved PDFs are available. Upload a PDF first.")
        embeddings = upload_index["embeddings"]
        chunks = upload_index["chunks"]
        bm25 = upload_index["bm25"]
    elif search_mode == "Built-in papers":
        embeddings = base_embeddings
        chunks = base_chunks
        bm25 = base_bm25
    else:
        if upload_index is not None:
            embeddings = np.vstack([base_embeddings, upload_index["embeddings"]])
            chunks = base_chunks + upload_index["chunks"]
            bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
        else:
            embeddings = base_embeddings
            chunks = base_chunks
            bm25 = base_bm25

    results = retrieve_chunks(question, embedding_model, embeddings, chunks, bm25)
    results = deduplicate_results(results)
    results = prioritize_paper_metadata(question, results, chunks)
    return results


# =========================================================
# 11. PROCESS ONE PENDING QUESTION
# =========================================================
def answer_pending_question(chat, pdf_records):
    """Answer the saved pending question; keep it pending if an error occurs."""
    question = chat.get("pending_question")
    if not question:
        return

    search_mode = chat.get("search_mode", "Uploaded PDFs")
    try:
        with st.spinner("Searching your research papers..."):
            results = search_documents(question, search_mode, pdf_records)

        if not results:
            answer = (
                "I couldn't retrieve relevant passages for this question. "
                "Try rephrasing it or selecting a different document collection."
            )
        else:
            prompt = build_app_prompt(question, results)
            with st.spinner("Generating your answer..."):
                answer = generate_answer(prompt)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("The answer model returned an empty response.")
            answer = answer.strip()

        chat["messages"].append({
            "role": "assistant",
            "content": answer,
            "evidence": results,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        chat["pending_question"] = None
        chat["pending_status"] = None
        chat["last_error"] = None
        save_chat(chat)
        st.rerun()

    except Exception as error:
        chat["pending_status"] = "failed"
        chat["last_error"] = str(error)
        save_chat(chat)
        st.error(f"Could not answer the pending question: {error}")
        st.warning("Your question has been saved. Retry it before submitting another question.")


# =========================================================
# 12. SIDEBAR: DOCUMENTS AND SAVED CHATS
# =========================================================
st.title("📚 Research Paper Assistant")
st.write("Ask questions about research papers, inspect retrieved evidence, and return to previous conversations anytime.")

with st.sidebar:
    st.header("📄 Documents")
    uploaded_files = st.file_uploader(
        "Add research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Uploaded files are saved locally for future sessions.",
    )
    try:
        persist_uploaded_files(uploaded_files)
        st.session_state.upload_error = None
    except Exception as error:
        st.session_state.upload_error = str(error)

    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)

    pdf_records = get_saved_pdfs()
    st.caption(f"Saved PDF files: {len(pdf_records)}")

    if pdf_records:
        with st.expander("Manage saved PDFs"):
            for filename, _pdf_bytes in pdf_records:
                col1, col2 = st.columns([4, 1])
                col1.write(filename)
                if col2.button(
                    "🗑️",
                    key=f"delete_pdf_{hashlib.sha256(filename.encode()).hexdigest()[:12]}",
                    help=f"Delete {filename}",
                ):
                    try:
                        (UPLOAD_DIR / filename).unlink()
                        st.session_state.upload_index = None
                        st.session_state.upload_fingerprint = None
                        st.rerun()
                    except OSError as error:
                        st.error(f"Could not delete PDF: {error}")

    st.divider()
    st.header("🔎 Search settings")
    active_chat = get_active_chat()
    initial_mode = active_chat.get("search_mode", "Uploaded PDFs") if active_chat else "Uploaded PDFs"
    if initial_mode not in SEARCH_MODES:
        initial_mode = "Uploaded PDFs"

    search_mode = st.selectbox(
        "Search collection",
        SEARCH_MODES,
        index=SEARCH_MODES.index(initial_mode),
        key=f"search_mode_{active_chat['id']}" if active_chat else "search_mode_new",
    )
    if active_chat and not active_chat.get("pending_question"):
        if active_chat.get("search_mode") != search_mode:
            active_chat["search_mode"] = search_mode
            save_chat(active_chat)

    st.caption(f"Embedding model: {EMBEDDING_MODEL}")
    st.caption(f"Answer model: {LLM_MODEL}")
    st.caption("Retrieval: Dense + BM25 + RRF")
    st.caption(f"Chunk settings: {CHUNK_SIZE} characters / {CHUNK_OVERLAP} overlap")
    st.divider()
    st.header("💬 Chat history")

    if st.button("＋ New Chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    chats = sorted(
        st.session_state.chats.values(),
        key=lambda item: item.get("updated_at", ""),
        reverse=True,
    )
    if not chats:
        st.caption("Your saved conversations will appear here.")

    for saved_chat in chats:
        chat_id = saved_chat["id"]
        is_active = chat_id == st.session_state.active_chat_id
        title = saved_chat.get("title", "New Chat")
        if saved_chat.get("pending_question"):
            title = "⏳ " + title
        button_label = f"● {title}" if is_active else title
        if st.button(
            button_label,
            key=f"open_chat_{chat_id}",
            use_container_width=True,
            help="Open this saved conversation",
        ):
            st.session_state.active_chat_id = chat_id
            st.rerun()


# =========================================================
# 13. SELECT OR CREATE AN ACTIVE CONVERSATION
# =========================================================
active_chat = get_active_chat()
if active_chat is None:
    st.info("Start a new chat or select an existing conversation from the sidebar.")
    st.button("＋ Start New Chat", on_click=start_new_chat)
    st.stop()


# =========================================================
# 14. DISPLAY THE CURRENT CONVERSATION
# =========================================================
st.subheader(f"💬 {active_chat['title']}")
st.caption(f"Created: {active_chat.get('created_at', 'Unknown')}")

if active_chat.get("pending_question"):
    st.warning("This conversation has an unanswered question. Finish answering it before asking another question.")

for message in active_chat.get("messages", []):
    with st.chat_message(message.get("role", "assistant")):
        st.markdown(message.get("content", ""))
        evidence = message.get("evidence", [])
        if evidence:
            with st.expander("🔍 Inspect retrieved evidence"):
                for item in evidence:
                    st.markdown(
                        f"**Source:** `{item.get('source', 'Unknown')}`  \n"
                        f"**Page:** {item.get('page', 'Unknown')}"
                    )
                    details = []
                    for field in ("dense_rank", "bm25_rank", "rrf_score"):
                        if field in item:
                            value = item[field]
                            if field == "rrf_score":
                                try:
                                    value = f"{float(value):.6f}"
                                except (TypeError, ValueError):
                                    pass
                            details.append(f"{field}: {value}")
                    if details:
                        st.caption(" | ".join(details))
                    st.text(item.get("text", ""))
                    st.divider()


# =========================================================
# 15. RETRY A FAILED QUESTION
# =========================================================
if active_chat.get("pending_question") and active_chat.get("pending_status") == "failed":
    st.error(active_chat.get("last_error", "The previous question could not be answered."))
    st.markdown("**Pending question:**")
    st.write(active_chat["pending_question"])
    if st.button("🔄 Retry unanswered question", use_container_width=True):
        active_chat["pending_status"] = "queued"
        active_chat["last_error"] = None
        save_chat(active_chat)
        st.rerun()


# =========================================================
# 16. PROCESS THE QUEUED QUESTION BEFORE ACCEPTING ANOTHER
# =========================================================
if active_chat.get("pending_question") and active_chat.get("pending_status") == "queued":
    answer_pending_question(active_chat, pdf_records)


# =========================================================
# 17. QUESTION INPUT
# =========================================================
question_disabled = bool(active_chat.get("pending_question"))
question = st.chat_input(
    "Ask a question about your research papers...",
    disabled=question_disabled,
)

if question:
    question = question.strip()
    if not question:
        st.warning("Please enter a question.")
        st.stop()
    if active_chat.get("pending_question"):
        st.warning("Please finish the previous question first.")
        st.stop()
    if not tokenize(question):
        st.warning("Please enter a question containing searchable words.")
        st.stop()

    if not active_chat["messages"]:
        active_chat["title"] = make_chat_title(question)
    active_chat["search_mode"] = search_mode
    active_chat["messages"].append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    active_chat["pending_question"] = question
    active_chat["pending_status"] = "queued"
    active_chat["last_error"] = None
    save_chat(active_chat)
    st.rerun()

