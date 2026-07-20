"""Query RAG sul vector store.

Uso:
  python query.py "Quali sono i setup di reversal piu' affidabili?"
  python query.py "position sizing per prop firm" --k 8
  python query.py "AMM impermanent loss" --no-llm      # solo retrieval

Con ANTHROPIC_API_KEY impostata, la risposta viene generata da Claude
usando i chunk recuperati come contesto (con citazioni file/pagina).
Senza API key, o con --no-llm, stampa i chunk recuperati.
"""
import argparse
import os
import sys

import chromadb
from fastembed import TextEmbedding

import config


def retrieve(question: str, k: int):
    model = TextEmbedding(config.EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR))
    collection = client.get_collection(config.COLLECTION_NAME)

    emb = [next(model.embed([config.QUERY_PREFIX + question])).tolist()]
    res = collection.query(query_embeddings=emb, n_results=k)

    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"text": doc, "meta": meta, "score": 1 - dist})
    return hits


def format_hits(hits):
    parts = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        pages = (
            f"p. {m['page_start']}"
            if m["page_start"] == m["page_end"]
            else f"pp. {m['page_start']}-{m['page_end']}"
        )
        parts.append(
            f"[{i}] {m['source']} ({pages}) — score {h['score']:.3f}\n{h['text']}"
        )
    return "\n\n---\n\n".join(parts)


def ask_claude(question, hits):
    import anthropic

    context = format_hits(hits)
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=2048,
        system=(
            "Sei un assistente esperto di trading. Rispondi SOLO in base agli "
            "estratti forniti. Cita sempre le fonti nel formato "
            "[nome_file, pagina]. Se gli estratti non contengono la risposta, dillo."
        ),
        messages=[
            {
                "role": "user",
                "content": f"ESTRATTI:\n\n{context}\n\nDOMANDA: {question}",
            }
        ],
    )
    return msg.content[0].text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--no-llm", action="store_true", help="solo retrieval")
    args = ap.parse_args()

    hits = retrieve(args.question, args.k)
    if not hits:
        sys.exit("Nessun risultato. Hai eseguito ingest.py?")

    if args.no_llm or not os.environ.get("ANTHROPIC_API_KEY"):
        if not args.no_llm:
            print("(ANTHROPIC_API_KEY non impostata: mostro solo il retrieval)\n")
        print(format_hits(hits))
    else:
        print(ask_claude(args.question, hits))
        print("\n--- Fonti ---")
        for h in hits:
            m = h["meta"]
            print(f"- {m['source']} pp. {m['page_start']}-{m['page_end']}")


if __name__ == "__main__":
    main()
