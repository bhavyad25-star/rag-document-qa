# Research Paper Assistant — RAG-Based Question Answering

A research-paper question-answering application that helps users explore research papers by asking natural-language questions and retrieving relevant passages from the documents.

The project implements a Retrieval-Augmented Generation (RAG) pipeline that combines information retrieval techniques with language-model-based answer generation. It also includes experiments to compare retrieval methods and evaluate how effectively relevant passages are retrieved.

## Features

- **Research paper question answering:** Ask questions about the papers in the document collection.
- **Dense retrieval:** Uses sentence embeddings to identify passages with similar semantic meaning.
- **BM25 retrieval:** Uses keyword-based ranking to find passages that share important terms with the query.
- **Hybrid retrieval:** Combines dense and BM25 rankings using Reciprocal Rank Fusion (RRF).
- **Retrieval evaluation:** Compares retrieval strategies using Success@K metrics.
- **Local development:** Supports running the Streamlit application on a local computer.
- **Hosted generation integration:** The project is being prepared to use a hosted LLM API for public deployment.

## Architecture

The application follows a retrieval-augmented generation workflow:

1. **Document collection:** Research papers are stored as PDF files.
2. **Text extraction and chunking:** Text is extracted from PDFs and divided into smaller passages.
3. **Embedding generation:** The passages are encoded into vector representations using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Retrieval:** Dense retrieval and BM25 identify relevant passages. Hybrid retrieval combines their rankings using Reciprocal Rank Fusion.
5. **Context construction:** Retrieved passages are prepared as context for the language model.
6. **Answer generation:** A language model generates an answer using the retrieved context.
7. **User interface:** Streamlit provides the interface for asking questions and viewing responses.

## Retrieval Methods

### 1. Dense Retrieval

Dense retrieval represents queries and document passages as numerical embeddings. Similarity scores are used to rank passages according to semantic relevance.

### 2. BM25 Retrieval

BM25 is a lexical ranking method that scores passages based on query-term matches, term frequency, and document-length normalization.

### 3. Hybrid Retrieval

Hybrid retrieval combines dense and BM25 rankings through Reciprocal Rank Fusion (RRF). This allows the system to use both semantic similarity and exact keyword matching.

### 4. Hybrid Retrieval with Reranking

The project also includes an experiment using a cross-encoder reranker to rescore candidate passages for relevance to a query.

## Evaluation

Retrieval performance is evaluated using **Success@K**, which measures the proportion of evaluation questions for which at least one relevant passage appears among the top K retrieved results.

The latest recorded results after correcting evaluation labels were:

| Retrieval method | Success@3 | Success@5 | Success@10 |
|---|---:|---:|---:|
| Dense retrieval | 66.67% | 80.00% | 100.00% |
| BM25 retrieval | 80.00% | 86.67% | 100.00% |
| Hybrid retrieval | 86.67% | 93.33% | 100.00% |

These results were obtained on a small evaluation set of 15 questions. They represent retrieval performance, not answer-generation accuracy. Results may change with different evaluation questions, relevance labels, document collections, or retrieval settings.

The experiments suggest that hybrid retrieval improved top-3 and top-5 retrieval success compared with either method alone in this evaluation.

## Technology Stack

- **Python** — application and retrieval logic
- **Streamlit** — interactive web interface
- **Sentence Transformers** — text embeddings
- **NumPy** — embedding storage and numerical operations
- **rank-bm25** — BM25 keyword retrieval
- **PyMuPDF** — PDF text extraction
- **Ollama** — local language-model inference during local development
- **Groq API** — planned hosted language-model inference for deployment

## Project Structure

```text
rag-document-qa/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   │   └── papers/
│   ├── processed/
│   │   ├── chunks.jsonl
│   │   ├── embedded_chunks.jsonl
│   │   ├── embeddings.npy
│   │   └── evaluation_questions.jsonl
│   └── backups/
├── experiments/
│   └── baseline_top3.txt
└── src/
    ├── answer_question.py
    ├── chunk_papers.py
    ├── embed_chunks.py
    ├── search_chunks.py
    ├── bm25_search.py
    ├── hybrid_search.py
    ├── evaluate_retrieval.py
    ├── compare_retrieval.py
    ├── compare_reranker.py
    └── other evaluation and diagnostic scripts
```

Some files may differ as the project evolves.

## Research Papers

The current document collection contains three research papers:

- *Attention Is All You Need* — Transformer architecture.
- *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — retrieval-augmented generation.
- *Dense Passage Retrieval for Open-Domain Question Answering* — dense passage retrieval.

## Local Setup

### Prerequisites

- Python 3.10 or later (Python 3.12 was used during development).
- Git.
- A compatible local language model through Ollama for the original local-generation setup, or a configured hosted LLM API after the deployment integration is completed.

### 1. Clone the repository

```bash
git clone https://github.com/bhavyad25-star/rag-document-qa.git
cd rag-document-qa
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the processed data

The application expects the processed chunks and embedding files used by the retrieval pipeline. If these files are not included in the repository, run the project's document-processing and embedding scripts to regenerate them.

### 5. Configure language-model access

For the original local setup, install Ollama and download the model configured in `src/answer_question.py`.

For hosted deployment, configure the required API key through the deployment platform's secrets settings. Never commit API keys or secret configuration files to GitHub.

### 6. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## Evaluation Scripts

The `src/` directory contains scripts for document processing, retrieval, evaluation, and comparison of retrieval strategies.

Run a script from the project root using Python. For example:

```bash
python src/evaluate_retrieval.py
```

Use the script's actual configuration and data requirements before running it. Evaluation results depend on the corpus, relevance labels, and retrieval parameters.

## Limitations and Future Improvements

- The current retrieval evaluation uses a small set of 15 questions and should be expanded for stronger conclusions.
- Retrieval Success@K does not directly measure answer correctness, factual consistency, or citation accuracy.
- Language-model answer quality depends on the relevance of retrieved passages and the generation model.
- Public deployment requires a reachable hosted LLM endpoint; a local Ollama URL will not work directly from a cloud-hosted application.
- Further improvements include expanding the evaluation set, testing chunk sizes and overlap, improving reranking, measuring answer faithfulness, and testing the deployed application.

## Future Work

- Expand the research-paper collection.
- Increase the size and diversity of the evaluation dataset.
- Compare retrieval configurations using consistent relevance labels.
- Evaluate answer correctness and faithfulness separately from retrieval success.
- Improve source and page citations in generated answers.
- Optimize startup time and cloud resource usage.

## Author

Developed as an academic AI/ML project exploring document retrieval, hybrid search, and Retrieval-Augmented Generation.

## Repository

[GitHub — bhavyad25-star/rag-document-qa](https://github.com/bhavyad25-star/rag-document-qa)
