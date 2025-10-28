import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv() 

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "sample.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Creating...")
    os.makedirs(persistent_directory, exist_ok=True)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("Document chunk information")
    print("Number of chunks:", len(docs))
    if len(docs) > 0:
        print("Sample chunk:", docs[0].page_content)

    print("Creating embedding and persisting to disk...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Finished embedding creation.")

    print("Creating vectors")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("Persisting vectors finished")
else:
    print("Persistent directory exists. Skipping embedding creation.")