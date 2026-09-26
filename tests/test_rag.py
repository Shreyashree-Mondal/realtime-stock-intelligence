"""RAG helpers that don't need a database or an LLM."""
from genai.agent import parse_route
from genai.rag import chunk_markdown, detect_symbol, to_pgvector

DOC = """# Title
Intro paragraph.

## 1. Kafka
Kafka text.

## 2. Spark
""" + ("Spark sentence. " * 150)


def test_chunks_follow_headings():
    titles = [t for t, _ in chunk_markdown(DOC, "doc")]
    assert titles[0] == "Title"
    assert "1. Kafka" in titles and "2. Spark" in titles


def test_long_sections_are_split_but_keep_their_title():
    spark_chunks = [c for t, c in chunk_markdown(DOC, "doc", max_chars=500) if t == "2. Spark"]
    assert len(spark_chunks) >= 1
    assert all(len(c) <= 2600 for c in spark_chunks)


def test_router_reads_json_even_with_extra_text():
    assert parse_route('Sure! {"tools": ["sql", "rag"]}') == ["sql", "rag"]


def test_router_falls_back_to_keywords():
    assert parse_route("I think you should explain why") == ["rag"]
    assert parse_route("garbage") == ["sql"]


def test_symbol_detection_handles_crypto_and_nse_tickers():
    syms = ["AAPL", "BINANCE:BTCUSDT", "TCS.NS"]
    assert detect_symbol("any news on aapl today?", syms) == "AAPL"
    assert detect_symbol("How risky is TCS?", syms) == "TCS.NS"
    assert detect_symbol("btcusdt volatility", syms) == "BINANCE:BTCUSDT"
    assert detect_symbol("general question", syms) is None


def test_vector_literal_format():
    assert to_pgvector([0.5, -1.0]) == "[0.500000,-1.000000]"
