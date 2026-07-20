# RAG Pipeline — TradingKnowledge

Pipeline di ingestione lossless dei PDF della cartella in un vector store ChromaDB, interrogabile via RAG con Claude.

## Come funziona

L'ingestione estrae il testo pagina per pagina con PyMuPDF, converte le tabelle in Markdown, applica OCR di fallback alle pagine scansionate (se pytesseract è installato) e salva l'intero testo estratto in `extracted/*.md` come archivio senza perdita. Il testo viene poi diviso in chunk (~1200 caratteri, overlap 250) rispettando i paragrafi, con metadati di file e pagine per le citazioni, e indicizzato in ChromaDB con embedding locali multilingue (`paraphrase-multilingual-MiniLM-L12-v2` via fastembed/ONNX, senza torch) — così puoi fare domande in italiano su testi in inglese.

L'ingestione è idempotente: rilanciandola, i PDF già indicizzati (stesso hash) vengono saltati; i PDF nuovi o modificati vengono (ri)indicizzati. Basta aggiungere PDF alla cartella e rilanciare `ingest.py`.

## Setup

```bash
cd rag_pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python ingest.py            # indicizza i PDF (incrementale)
python ingest.py --force    # re-indicizza tutto

python query.py "Quali setup di reversal hanno il win rate migliore?"
python query.py "position sizing" --k 8 --no-llm   # solo retrieval, senza LLM
```

Per le risposte generate da Claude serve una API key. Due modi, a scelta:

1. `cp .env.example .env` e incolla la tua chiave nel file `.env` (viene caricato automaticamente, non serve altro).
2. In alternativa, `export ANTHROPIC_API_KEY=sk-ant-...` nel terminale (o nel tuo `~/.zshrc` per renderlo permanente).

Senza chiave, `query.py` e `signal_check.py` mostrano solo i chunk recuperati, senza generare la risposta.

Modello usato: `claude-sonnet-5` (configurabile in `config.py`, variabile `CLAUDE_MODEL`).

## Analisi tecnica live + RAG

`signal_check.py` scarica gli OHLCV via ccxt (Binance di default, senza API key), calcola ~50 indicatori (RSI, MACD, EMA/SMA 20-50-200, ADX, Stocastico, StochRSI, CCI, Williams %R, ROC, Bollinger, Keltner, Donchian, ATR, Ichimoku, PSAR, MFI, OBV, CMF, VWAP...), li interpreta e chiede al RAG cosa direbbero i tuoi libri su quello scenario:

```bash
python signal_check.py BTC                    # BTC/USDT, timeframe 4h
python signal_check.py BTC ETH SOL --tf 1d
python signal_check.py BTC --no-llm           # solo dashboard + passaggi dai libri
python signal_check.py BTC --exchange kraken --quote USD
```

Strumento informativo, non consulenza finanziaria.

## Struttura

```
rag_pipeline/
├── config.py             # tutti i parametri (modello, chunking, percorsi)
├── ingest.py             # PDF → estrazione → chunk → embedding → ChromaDB
├── query.py              # retrieval + risposta RAG con citazioni
├── extracted/            # testo integrale estratto (archivio lossless, .md)
├── vector_store/         # ChromaDB persistente
└── ingest_manifest.json  # hash dei file già ingeriti
```

## Note

Per la massima qualità di embedding, in `config.py` imposta `EMBEDDING_MODEL = "intfloat/multilingual-e5-large"` (~2.2 GB, più lento) e rilancia `ingest.py --force`. Il vector store si usa da qualsiasi codice Python/LangChain puntando ChromaDB a `vector_store/`, collezione `trading_knowledge`.
