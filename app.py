"""
HR Policy Assistant — RAG (Retrieval-Augmented Generation)
------------------------------------------------------------
A beginner-friendly Streamlit app that lets a user upload an HR Policy PDF
and ask questions about it. The app:

1. Extracts text from the PDF using PyMuPDF (fitz)
2. Cleans and splits the text into overlapping chunks
3. Generates embeddings for each chunk using Sentence Transformers
4. Stores the embeddings in a FAISS index for fast similarity search
5. Finds the most relevant chunks for a user's question
6. Sends the question + relevant chunks to Groq (model: openai/gpt-oss-20b)
7. Displays the answer, and tells the user clearly if the answer is not
   found in the uploaded PDF (instead of making something up).
"""

import os
import re

import fitz  # PyMuPDF
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# ----------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📄",
    layout="wide",
)

GROQ_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ----------------------------------------------------------------------
# CACHED RESOURCES
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """
    Load the Sentence Transformer embedding model once and reuse it.

    'all-MiniLM-L6-v2' is chosen because it is:
    - Small (~80MB), so it loads quickly on Streamlit Cloud's limited resources
    - Fast at generating embeddings, even on CPU-only servers
    - Reliable and well-tested for general-purpose semantic similarity tasks
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_groq_client():
    """
    Create a Groq client using the API key stored securely in Streamlit
    Cloud Secrets. The key is never hard-coded in this file.
    """
    api_key = st.secrets.get("GROQ_API_KEY", None) or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ----------------------------------------------------------------------
# PDF PROCESSING
# ----------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    """
    Extract text from every page of the uploaded PDF using PyMuPDF.
    Returns a list of dicts: [{"page": page_number, "text": page_text}, ...]

    Pages with little or no extractable text (e.g. scanned images) are
    still included with an empty string, so page numbers stay accurate,
    but they are filtered out later before chunking.
    """
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text("text") or ""
        pages.append({"page": page_number + 1, "text": text})

    doc.close()
    return pages


def clean_text(text):
    """
    Basic text cleaning:
    - Collapse multiple whitespace/newlines into single spaces
    - Strip leading/trailing whitespace
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------------------------------------------------------------------
# CHUNKING
# ----------------------------------------------------------------------
def chunk_text(pages, chunk_size=200, chunk_overlap=40):
    """
    Split page text into overlapping word-based chunks.

    chunk_size:    number of words per chunk
    chunk_overlap: number of words repeated between consecutive chunks

    Overlap is used so that a sentence or idea that gets cut off at the
    boundary of one chunk is not lost — it also appears (partially or
    fully) at the start of the next chunk. This helps the retrieval step
    find relevant information even when it sits right on a chunk boundary.

    Returns a list of dicts: [{"text": chunk_text, "page": page_number}, ...]
    """
    chunks = []

    for page_info in pages:
        page_number = page_info["page"]
        text = clean_text(page_info["text"])

        if not text:
            continue  # skip pages with no extractable text

        words = text.split(" ")
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_str = " ".join(chunk_words).strip()

            if chunk_str:
                chunks.append({"text": chunk_str, "page": page_number})

            if end == len(words):
                break

            # Move the window forward, leaving `chunk_overlap` words behind
            start = end - chunk_overlap
            if start < 0:
                start = 0

    return chunks


# ----------------------------------------------------------------------
# EMBEDDINGS + FAISS INDEX
# ----------------------------------------------------------------------
def build_faiss_index(chunks, embedding_model):
    """
    Generate embeddings for all chunks and build a FAISS index.

    We normalize embeddings and use an Inner Product (IP) index, which is
    mathematically equivalent to cosine similarity search on normalized
    vectors. This is a common, simple, and effective approach for
    semantic search.

    Returns: (faiss_index, embeddings_array)
    """
    texts = [c["text"] for c in chunks]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype("float32")

    # Normalize each vector to unit length for cosine-similarity via inner product
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def search_index(question, embedding_model, index, chunks, top_k=4):
    """
    Embed the user's question, search the FAISS index, and return the
    top_k most relevant chunks (with their similarity scores).
    """
    if index is None or not chunks:
        return []

    question_embedding = embedding_model.encode(
        [question],
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(question_embedding)

    top_k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "score": float(score),
        })
    return results


# ----------------------------------------------------------------------
# GROQ LLM CALL
# ----------------------------------------------------------------------
def build_prompt(question, retrieved_chunks):
    """
    Build the prompt sent to the LLM. The instructions explicitly force
    the model to rely only on the supplied context and to admit when the
    answer is not present, instead of inventing policy details.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(f"[Chunk {i} | Page {chunk['page']}]\n{chunk['text']}")
    context_text = "\n\n".join(context_blocks) if context_blocks else "No context available."

    system_prompt = (
        "You are an HR Policy Assistant. You answer questions strictly using "
        "the HR policy context provided below.\n"
        "Rules you MUST follow:\n"
        "1. Use ONLY the information in the provided context to answer.\n"
        "2. Do NOT invent, assume, or add any policy information that is not "
        "explicitly present in the context.\n"
        "3. If the context does not contain the answer, respond clearly with: "
        "\"I could not find this information in the uploaded HR policy.\"\n"
        "4. Keep answers concise, clear, and easy to understand.\n"
        "5. When useful, mention which page number the information came from, "
        "based on the context labels (e.g. 'Page 3')."
    )

    user_prompt = (
        f"HR Policy Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above."
    )

    return system_prompt, user_prompt


def generate_answer(question, retrieved_chunks, client):
    """
    Send the question and retrieved context to Groq's chat completion
    endpoint using the openai/gpt-oss-20b model, and return the answer text.
    """
    system_prompt, user_prompt = build_prompt(question, retrieved_chunks)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


# ----------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ----------------------------------------------------------------------
def init_session_state():
    defaults = {
        "chunks": [],
        "faiss_index": None,
        "pdf_name": None,
        "pdf_processed": False,
        "chat_history": [],  # list of {"question": ..., "answer": ..., "sources": [...]}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_app():
    st.session_state["chunks"] = []
    st.session_state["faiss_index"] = None
    st.session_state["pdf_name"] = None
    st.session_state["pdf_processed"] = False
    st.session_state["chat_history"] = []


# ----------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------
def main():
    init_session_state()

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("📄 Upload HR Policy")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        st.subheader("Retrieval Settings")
        top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=4)

        with st.expander("Advanced: Chunking Settings"):
            chunk_size = st.number_input(
                "Chunk size (words)", min_value=50, max_value=500, value=200, step=10
            )
            chunk_overlap = st.number_input(
                "Chunk overlap (words)", min_value=0, max_value=200, value=40, step=10
            )

        st.divider()
        if st.button("🔄 Clear / Reset"):
            reset_app()
            st.rerun()

        st.divider()
        st.subheader("ℹ️ How RAG Works")
        st.markdown(
            "1. Your PDF is split into small text chunks.\n"
            "2. Each chunk is converted into a numeric 'embedding'.\n"
            "3. When you ask a question, it's also converted into an embedding.\n"
            "4. FAISS finds the chunks whose embeddings are most similar "
            "to your question.\n"
            "5. Those chunks are sent to the AI model as context, so it can "
            "answer using only your document."
        )

    # ---------------- MAIN AREA ----------------
    st.title("🧭 HR Policy Assistant")
    st.write(
        "Upload your company's HR Policy PDF, then ask questions about it. "
        "This assistant uses **Retrieval-Augmented Generation (RAG)** to find "
        "the most relevant sections of your policy and answer using only that "
        "information."
    )

    embedding_model = load_embedding_model()

    # Process the uploaded PDF (only re-process if it's a new file)
    if uploaded_file is not None:
        is_new_file = st.session_state["pdf_name"] != uploaded_file.name
        if is_new_file or not st.session_state["pdf_processed"]:
            with st.spinner("📖 Reading and processing your PDF..."):
                try:
                    pages = extract_text_from_pdf(uploaded_file)
                    total_text_length = sum(len(p["text"].strip()) for p in pages)

                    if total_text_length == 0:
                        st.error(
                            "⚠️ No extractable text was found in this PDF. "
                            "It may be a scanned document (images only). "
                            "Please upload a PDF that contains selectable text."
                        )
                    else:
                        chunks = chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                        if not chunks:
                            st.error("⚠️ Could not create any text chunks from this PDF.")
                        else:
                            index, _ = build_faiss_index(chunks, embedding_model)

                            st.session_state["chunks"] = chunks
                            st.session_state["faiss_index"] = index
                            st.session_state["pdf_name"] = uploaded_file.name
                            st.session_state["pdf_processed"] = True
                            st.session_state["chat_history"] = []

                            st.success(
                                f"✅ Processed **{uploaded_file.name}** — "
                                f"{len(pages)} page(s), {len(chunks)} chunk(s) created."
                            )
                except Exception as e:
                    st.error(f"❌ Failed to process the PDF: {e}")

    # Status banner
    if st.session_state["pdf_processed"]:
        st.info(f"📌 Currently loaded policy document: **{st.session_state['pdf_name']}**")
    else:
        st.warning("👆 Please upload an HR Policy PDF from the sidebar to get started.")

    st.divider()

    # ---------------- QUESTION INPUT ----------------
    st.subheader("💬 Ask a Question")
    question = st.text_input(
        "Type your question about the HR policy",
        placeholder="e.g. How many annual leave days am I entitled to?",
        disabled=not st.session_state["pdf_processed"],
    )

    ask_clicked = st.button("Ask", type="primary", disabled=not st.session_state["pdf_processed"])

    if ask_clicked:
        if not question or not question.strip():
            st.warning("Please type a question before clicking Ask.")
        else:
            client = get_groq_client()
            if client is None:
                st.error(
                    "❌ Groq API key not found. Please add `GROQ_API_KEY` to your "
                    "Streamlit Secrets (see README for instructions)."
                )
            else:
                with st.spinner("🔍 Searching the policy document..."):
                    retrieved = search_index(
                        question,
                        embedding_model,
                        st.session_state["faiss_index"],
                        st.session_state["chunks"],
                        top_k=top_k,
                    )

                with st.spinner("🤖 Generating answer..."):
                    answer, error = generate_answer(question, retrieved, client)

                if error:
                    st.error(f"❌ Error while calling Groq API: {error}")
                else:
                    st.session_state["chat_history"].insert(0, {
                        "question": question,
                        "answer": answer,
                        "sources": retrieved,
                    })

    # ---------------- ANSWER / HISTORY DISPLAY ----------------
    if st.session_state["chat_history"]:
        st.divider()
        st.subheader("📝 Answers")

        for i, entry in enumerate(st.session_state["chat_history"]):
            st.markdown(f"**Q: {entry['question']}**")
            st.markdown(entry["answer"])

            with st.expander("📚 View retrieved policy sources"):
                if entry["sources"]:
                    for j, src in enumerate(entry["sources"], start=1):
                        st.markdown(
                            f"**Source {j} — Page {src['page']} "
                            f"(similarity score: {src['score']:.3f})**"
                        )
                        st.text(src["text"])
                        st.markdown("---")
                else:
                    st.write("No sources were retrieved for this question.")

            st.divider()


if __name__ == "__main__":
    main()
