# Financial RAG System

Система RAG для анализа финансовых отчетов. Включает полный цикл: от загрузки PDF до отправки ответов.

## Структура

*   `src/ingestion.py`: Обработка PDF через Docling, OCR, извлечение названий компаний через LLM.
*   `src/indexing.py`: Создание векторного индекса FAISS с учетом ограничений памяти GPU.
*   `src/retrieval.py`: Гибридный поиск (BM25 + Vector) с фильтрацией по метаданным.
*   `src/generation.py`: Генерация ответов через Ollama.

## Установка

1.  **Требования**: Python 3.10+, [uv](https://astral.sh/uv), Ollama, GPU (рекомендуется).
2.  **Зависимости**:
    ```bash
    uv sync
    ```
3.  **Системные настройки**:
    ```bash
    uv run main.py --setup
    # В отдельном терминале запустите Ollama:
    ollama serve
    ```

## Данные

Поместите ваши PDF файлы в папку: `data/pdfs/`.
Поместите файл с вопросами в: `data/questions.json`.

## Использование

Весь процесс управляется через `src/main.py`.

### 1. Обработка PDF (Ingestion)
Конвертирует PDF в чанки и извлекает метаданные. Результат сохраняется в `data/new_splits.pickle`.
```bash
uv run -m src.main --ingest