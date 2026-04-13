from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_retriever():
    with open("data/sample.txt", "r") as f:
        text = f.read()

    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(docs, embeddings)

    return db