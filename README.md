# python-information-retrieval-system-mcp-
A modular Python information retrieval system implementing TF-IDF ranking, fuzzy matching, synonym expansion, and semantic reranking, inspired by modern AI search pipelines.

# Python Information Retrieval System

A modular text retrieval engine built in Python, combining multiple ranking and search techniques inspired by modern AI retrieval pipelines.

## Overview

This project implements a flexible information retrieval system capable of processing text queries and returning relevant results using a combination of lexical and semantic techniques.

The system is designed with a modular architecture, allowing different retrieval and ranking strategies to be combined and extended.

## Features

- TF-IDF based ranking
- Fuzzy string matching
- Synonym expansion
- Semantic re-ranking
- Modular search pipeline
- Server-based query handling

## Architecture

Query → Processing → Retrieval → Ranking → Results

The system integrates multiple engines:
- Lexical retrieval (TF-IDF)
- Approximate matching (fuzzy search)
- Query expansion (synonyms)
- Semantic-style re-ranking

## Project Structure

```text
.
├── demos/
├── experiments/
├── helpers/
├── .gitignore
├── README.md
├── adapter_stdio.py
├── http_facade.py
├── mcp_local.py
└── mcp_server.py
  

## Technologies

- Python
- Natural Language Processing (NLP)
- Information Retrieval
- Text Processing

## How to Run

Example:

```bash
python mcp_server.py
```markdown

## Example Use Cases

- Running local retrieval tests
- Evaluating ranking behavior across different query strategies
- Experimenting with fuzzy, lexical, and semantic-style retrieval pipelines

