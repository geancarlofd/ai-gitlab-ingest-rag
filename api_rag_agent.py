from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

app = FastAPI()

# Carrega o banco vetorial já salvo
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)

qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama3"),
    retriever=db.as_retriever()
)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    answer = qa_chain.run(query.question)
    return {"answer": answer}
