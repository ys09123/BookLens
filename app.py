import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import shutil
import gc
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

import warnings
warnings.filterwarnings("ignore", message=".*torchvision.*")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="BookLens")

st.title("📚 BookLens")
st.write("Upload a PDF and ask questions from the document")

# ── Session state ──────────────────────────────────────────────────────────
if "vectorstore"  not in st.session_state: st.session_state.vectorstore  = None
if "retriever"    not in st.session_state: st.session_state.retriever    = None
if "db_ready"     not in st.session_state: st.session_state.db_ready     = False
if "current_file" not in st.session_state: st.session_state.current_file = None

CHROMA_DIR = "./chroma_db"

# ── Cache embedding model so it loads only once ────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="/tmp/hf_cache",
    )

# ── Windows-safe Chroma deletion ───────────────────────────────────────────
def safe_delete_chroma():
    if st.session_state.vectorstore is not None:
        try:
            st.session_state.vectorstore._client.close()
        except Exception:
            pass
        st.session_state.vectorstore = None
        st.session_state.retriever   = None
        gc.collect()
        time.sleep(0.5)

    if os.path.exists(CHROMA_DIR):
        for attempt in range(5):
            try:
                shutil.rmtree(CHROMA_DIR)
                break
            except PermissionError:
                time.sleep(0.5 * (attempt + 1))
        else:
            raise RuntimeError(
                "Could not delete old database. Please restart the app and try again."
            )

# ── Upload ─────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    # Reset when a new file is chosen
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.db_ready     = False
        st.session_state.current_file = uploaded_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):
        # FIX 1: .pdf suffix so PyPDFLoader parses correctly
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Processing document..."):
                # FIX 2: wipe stale DB before building fresh
                safe_delete_chroma()

                loader = PyPDFLoader(tmp_path)
                docs   = loader.load()

                if not docs:
                    st.error("Could not extract text. Is the PDF scanned / image-only?")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,   # larger chunks = more context per retrieval
                    chunk_overlap=300, # more overlap = less chance of splitting an answer
                )
                chunks = splitter.split_documents(docs)

                embeddings = get_embeddings()

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=CHROMA_DIR,
                )

                # FIX 3: store in session_state so it survives reruns
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever   = vectorstore.as_retriever(
                    search_type="similarity",   # mmr hurts relevance at lambda=0.5
                    search_kwargs={"k": 6},     # fetch more chunks for wider coverage
                )
                st.session_state.db_ready = True

            st.success(f"Vector database created! ({len(docs)} pages · {len(chunks)} chunks)")

        except Exception as e:
            st.error(f"Error building database: {e}")
        finally:
            os.unlink(tmp_path)

# ── Q&A ────────────────────────────────────────────────────────────────────
if st.session_state.db_ready and st.session_state.retriever:

    llm = ChatMistralAI(model="mistral-small-2506")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ])

    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if query:
        with st.spinner("Searching document..."):
            retrieved_docs = st.session_state.retriever.invoke(query)

        if not retrieved_docs:
            st.warning("No relevant passages found. Try rephrasing your question.")
        else:
            context = "\n\n".join([d.page_content for d in retrieved_docs])
            final_prompt = prompt.invoke({"context": context, "question": query})
            response = llm.invoke(final_prompt)

            st.write("### AI Answer")
            st.write(response.content)

            with st.expander("🔍 Debug — retrieved chunks"):
                for i, d in enumerate(retrieved_docs, 1):
                    page = d.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i} · Page {page}**")
                    st.text(d.page_content)
                    st.divider()