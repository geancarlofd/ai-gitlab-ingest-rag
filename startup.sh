#!/bin/bash

# Inicia o serviço Ollama em segundo plano
ollama serve &

# Aguarda até o Ollama estar pronto
until ollama list > /dev/null 2>&1; do
  echo "Aguardando Ollama subir..."
  sleep 1
done

# Puxa o modelo de embedding
ollama pull nomic-embed-text

# Opcional: já inicializa o modelo principal
ollama run llama3

# Mantém o container vivo
tail -f /dev/null
