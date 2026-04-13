import streamlit as st
from utils.retriever import load_retriever
from utils.llm import generate_answer
from utils.web_loader import load_website

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="RAG Assistant", page_icon="📄")

st.title("📄 RAG Document Assistant")

# ---------- LOAD OR CREATE VECTOR DB ----------
@st.cache_resource
def load_db():
    try:
        return FAISS.load_local("vector_store", HuggingFaceEmbeddings())
    except:
        return load_retriever()

if "db" not in st.session_state:
    st.session_state.db = load_db()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Options")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.session_state.messages:
        chat_text = ""
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_text += f"{role}: {msg['content']}\n\n"

        st.download_button(
            "📥 Download Chat",
            data=chat_text,
            file_name="rag_chat.txt"
        )

# ---------- WEBSITE INPUT ----------
url = st.text_input("🌐 Enter website URL")

if url:
    text = load_website(url)

    if not text or len(text.strip()) < 50:
        st.error("❌ Unable to extract content from this website. Try another URL.")
    else:
        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        docs = splitter.split_text(text)

        if not docs:
            st.error("❌ No valid content found after processing.")
        else:
            embeddings = HuggingFaceEmbeddings()
            db = FAISS.from_texts(docs, embeddings)

            db.save_local("vector_store")

            st.session_state.db = db
            st.success("✅ Website content loaded & saved!")

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("📂 Upload your document", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    if not text or len(text.strip()) < 10:
        st.error("❌ File is empty or invalid.")
    else:
        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        docs = splitter.split_text(text)

        if not docs:
            st.error("❌ No usable content found in file.")
        else:
            embeddings = HuggingFaceEmbeddings()
            db = FAISS.from_texts(docs, embeddings)

            db.save_local("vector_store")

            st.session_state.db = db
            st.success("✅ Document loaded & saved!")

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- DISPLAY CHAT ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- INPUT ----------
user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            db = st.session_state.db

            docs = db.similarity_search(user_input)
            context = " ".join([doc.page_content for doc in docs])

            # Multi-turn memory
            history = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]]
            )

            full_prompt = f"""
You are an expert in call center analytics.

Use the given context and conversation history to answer.

Context:
{context}

Conversation History:
{history}

Question:
{user_input}

Instructions:
- Give clear answer
- Then explanation
- Then example if possible
"""

            response = generate_answer(full_prompt, user_input)

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()