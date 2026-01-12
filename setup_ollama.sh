### 10. setup_ollama.sh (Опциональный хелпер)

```bash
#!/bin/bash
# Простой скрипт для запуска Ollama в фоне и проверки модели

echo "Starting Ollama..."
nohup ollama serve > ollama.log 2>&1 &
PID=$!
echo "Ollama started with PID $PID"

sleep 5

echo "Pulling model gpt-oss:20b..."
ollama pull gpt-oss:20b

echo "Ready."