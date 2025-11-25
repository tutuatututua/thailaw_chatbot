from langchain_huggingface  import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
)
import os
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from datasets import load_dataset



# 4) Store vector DB

def create_vector_db():
    print("Loading dataset...")
    ds = load_dataset("pythainlp/thailaw")

    # Convert dataset rows to LangChain Docs
    docs = [
        Document(
            page_content=row["txt"],
            metadata={"sysid": row["sysid"], "title": row["title"]}
        )
        for row in ds["train"]
    ]

    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=[
        r"(?=มาตรา\s*\d+)",   # split before "มาตรา X"
        r"(?=ข้อ\s*\d+)",     # split before "ข้อ X"
        # Split before "ข้อ" + number (Arabic or Thai digits)
        r"(?=ข้อ\s*\n?\s*[๐-๙0-9]+)",

        # Paragraph break
        "\n\n",

        # Line break
        "\n",

        # Last fallback
        " ",
    ]
)

    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")
    embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

    print("Loading embeddings...")
    

    print("Creating Chroma vector database...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("Chroma DB created successfully!")

if __name__ == "__main__":
    create_vector_db()
