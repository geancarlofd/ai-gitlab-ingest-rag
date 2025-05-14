from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

db = Chroma(persist_directory="./chroma_db", embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
retriever = db.as_retriever()

llm = Ollama(model="llama3")  # ou outro modelo compatível com RAG

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

while True:
    query = input("Pergunta: ")
    result = qa(query)
    print("Resposta:", result["result"])
