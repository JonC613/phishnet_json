# Phish RAG Chatbot

This repository bundles a simple pipeline for exploring historical Phish setlists with Retrieval-Augmented Generation (RAG).

## Workflow

1. **Fetch data** – use the existing `fetch_phish_shows.py` script or drop JSON files into the `shows/` directory.
2. **Build the vector store** – embed and persist the shows locally:

   ```bash
   python ingest.py --shows-dir shows --persist-dir vectorstore --reset
   ```

3. **Chat with the data** – launch the assistant once the store exists:

   ```bash
   python phish_rag_app.py chat --persist-dir vectorstore --model gpt-4o-mini
   ```

Environment variables (`OPENAI_API_KEY`, `PHISHNET_API_KEY`) can be stored in `.env` for convenience.

## Commands

- `python ingest.py` – rebuild the Chroma vector store from downloaded shows. Supports chunk-size tuning and collection overrides.
- `python phish_rag_app.py chat` – open an interactive CLI chatbot; add `--show-context` to inspect retrieved passages.
- `python phish_rag_app.py validate` – print quick statistics about the locally cached shows.

Feel free to continue using the legacy scripts (`fetch_phish_show_single.py`, `create_new_vector_db.py`) if you still rely on the FAISS-based workflow.
