#!/usr/bin/env python3
"""Build and persist the vector store used by the Phish RAG chatbot."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from phishbot.data import build_documents, load_shows


DEFAULT_PERSIST_DIR = "vectorstore"
DEFAULT_COLLECTION = "phish-shows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or refresh the Phish vector store")
    parser.add_argument("shows_dir", nargs="?", default="shows", help="Directory with downloaded show JSON files")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Where to persist the Chroma database")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name to use")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--chunk-size", type=int, default=900, help="Character length for each chunk prior to embedding")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Overlap characters between chunks")
    parser.add_argument("--limit", type=int, help="Optional cap on number of shows to ingest (useful for tests)")
    parser.add_argument("--reset", action="store_true", help="Delete the existing vector store before ingestion")
    return parser.parse_args()


def ensure_reset(directory: Path, reset: bool) -> None:
    if reset and directory.exists():
        shutil.rmtree(directory)


def main() -> None:
    args = parse_args()
    load_dotenv()

    persist_dir = Path(args.persist_dir)
    ensure_reset(persist_dir, args.reset)

    shows = load_shows(args.shows_dir, limit=args.limit)
    if not shows:
        raise SystemExit("No shows found to ingest. Run fetch_phish_shows.py first.")

    print(f"Loaded {len(shows)} shows from {args.shows_dir}")
    documents = build_documents(shows)
    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunked_documents = splitter.split_documents(documents)
    for idx, doc in enumerate(chunked_documents):
        doc.metadata["chunk_id"] = idx

    print(f"Prepared {len(chunked_documents)} document chunks for embedding")

    embeddings = OpenAIEmbeddings(model=args.embedding_model)
    vectorstore = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        collection_name=args.collection,
        persist_directory=str(persist_dir),
    )
    vectorstore.persist()

    print("Vector store ready!")
    print(f"Location: {persist_dir}")
    print(f"Collection: {args.collection}")


if __name__ == "__main__":
    main()
