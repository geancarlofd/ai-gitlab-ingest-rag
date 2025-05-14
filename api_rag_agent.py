from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente do .env (opcional)
load_dotenv()

# Configurações (você pode alterar para usar variáveis de ambiente se quiser)
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL_EMBED = os.getenv("OLLAMA_MODEL_EMBED", "nomic-embed-text")
OLLAMA_MODEL_LLM = os.getenv("OLLAMA_MODEL_LLM", "llama3")

# Inicializa o FastAPI
app = FastAPI(title="RAG API com Ollama e ChromaDB")

# Inicializa o banco vetorial
try:
    db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=OllamaEmbeddings(model=OLLAMA_MODEL_EMBED, base_url=OLLAMA_URL)
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=Ollama(model=OLLAMA_MODEL_LLM, base_url=OLLAMA_URL),
        retriever=db.as_retriever()
    )
except Exception as e:
    raise RuntimeError(f"Erro ao inicializar o vetor ou o modelo: {e}")

# Modelo de entrada
class Query(BaseModel):
    question: str

# Endpoint de consulta
@app.post("/ask")
async def ask(query: Query):
    try:
        answer = qa_chain.run(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
