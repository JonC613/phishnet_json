#!/usr/bin/env python3
"""Interactive RAG chatbot for exploring Phish show history."""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from phishbot.data import load_shows, summarize_shows


DEFAULT_PERSIST_DIR = "vectorstore"
DEFAULT_COLLECTION = "phish-shows"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PhishBot RAG assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chatbot session")
    chat_parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Directory containing the Chroma vector store")
    chat_parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    chat_parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model")
    chat_parser.add_argument("--embedding-model", default="text-embedding-3-small", help="Embedding model for the retriever")
    chat_parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    chat_parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve for each answer")
    chat_parser.add_argument("--show-context", action="store_true", help="Print the retrieved context after each answer")

    validate_parser = subparsers.add_parser("validate", help="Inspect downloaded show data")
    validate_parser.add_argument("--shows-dir", default="shows", help="Directory with show JSON files")
    validate_parser.add_argument("--limit", type=int, help="Optional limit for quick checks")

    return parser


def format_context(documents) -> str:
    segments = []
    for idx, doc in enumerate(documents, 1):
        meta = doc.metadata
        location = ", ".join(
            filter(None, [meta.get("city"), meta.get("state"), meta.get("country")])
        )
        header = f"[{idx}] {meta.get('showdate', 'Unknown date')} - {meta.get('venue', 'Unknown venue')}"
        if location:
            header += f" ({location})"
        segments.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(segments)


def print_sources(documents) -> None:
    if not documents:
        return
    print("\nSources:")
    for idx, doc in enumerate(documents, 1):
        meta = doc.metadata
        location = ", ".join(
            filter(None, [meta.get("city"), meta.get("state"), meta.get("country")])
        )
        label = f"[{idx}] {meta.get('showdate', 'Unknown')} - {meta.get('venue', 'Unknown venue')}"
        if location:
            label += f" ({location})"
        print(f"  {label}")


def run_chat(args: argparse.Namespace) -> None:
    load_dotenv()

    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        raise SystemExit(
            f"Vector store not found at {persist_dir}. Run `python ingest.py` before chatting."
        )

    embeddings = OpenAIEmbeddings(model=args.embedding_model)
    vectorstore = Chroma(
        collection_name=args.collection,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})

    llm = ChatOpenAI(model=args.model, temperature=args.temperature)
    prompt = ChatPromptTemplate.from_template(
        textwrap.dedent(
            """
            You are PhishBot, a helpful assistant specializing in the live history of the band Phish.
            Use ONLY the provided context to answer questions. If the context lacks the answer, say so.
            Cite the relevant show date and venue inside square brackets like [1999-07-24 - Alpine Valley].

            Context:
            {context}

            Question: {question}
            """
        ).strip()
    )
    answer_chain = prompt | llm | StrOutputParser()

    print("\nWelcome to PhishBot! Ask about specific shows, songs, or tours.")
    print("Type 'exit' or 'quit' to finish.\n")

    while True:
        user_input = input("PhishBot> ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        start = time.time()
        documents = retriever.get_relevant_documents(user_input)
        if not documents:
            print("I couldn't retrieve any relevant shows. Try rephrasing your question.")
            continue

        context = format_context(documents)
        answer = answer_chain.invoke({"context": context, "question": user_input})
        elapsed = time.time() - start

        print("\n" + answer.strip())
        print(f"\n(answered in {elapsed:.2f}s)")
        if args.show_context:
            print("\nRetrieved context:\n")
            print(context)
        else:
            print_sources(documents)


def run_validate(args: argparse.Namespace) -> None:
    shows = load_shows(args.shows_dir, limit=args.limit)
    stats = summarize_shows(shows)

    if not stats["total_shows"]:
        print("No shows available. Use fetch_phish_shows.py to download data.")
        return

    print(f"Total shows: {stats['total_shows']}")
    print(f"Shows with venue info: {stats['shows_with_venue']}")
    print(f"Shows with setlists: {stats['shows_with_sets']}")
    print(f"Total songs captured: {stats['total_songs']}")
    print(f"Unique venues: {stats['unique_venues']}")
    print("Shows by year:")
    for year, count in sorted(stats["years"].items()):
        print(f"  {year}: {count}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "chat":
        run_chat(args)
    elif args.command == "validate":
        run_validate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
