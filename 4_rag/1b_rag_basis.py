import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv() 

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "What is Evershade?"

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k": 5, "score_threshold": 0.7},
)

relevant_docs = retriever.invoke(query)

print("Relevant document information")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:")
    print(doc.page_content)
    if doc.metadata:
        print("Source metadata:", doc.metadata.get("source", "N/A"))