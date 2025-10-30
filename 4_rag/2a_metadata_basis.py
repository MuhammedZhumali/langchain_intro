import os

from langchain_community.document_loaders import TextLoader
from langchain_text.splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persist_directory = os.path.join(db_dir, "chroma_metadata_basis")

print(f"Books directory: {books_dir}")
print(f"Persistant directory: {persist_directory}")

if not os.path.exists(persist_directory):
    print("Persistant directory does not exist. Creating...")

    if not os.path.exists(books_dir):
        raise FileNotFoundError{
            f"The directory {books_dir} does not exist."
        }
    
    books_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]

    documents = []

    for book_file in books_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("Number of document chunks:", len(docs))

    print("Creating embeddings...")
    embeddings =  OpenAIEmbeddings(
        model = "text-embedding-3-small"
    )
    print("Finished embeddings")

    print("Creating vector store")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persist_directory
    )
    print("Persisting vector store to disk")

else:
    print("Persistant directory already exists. Skipping vector store creation.")
