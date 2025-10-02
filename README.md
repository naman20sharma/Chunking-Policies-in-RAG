# Retrieval-Augmented Generation with Advanced Chunking

This project explores advanced document chunking strategies (Late Chunking, Meta-Chunking, Contextual Retrieval) to improve Retrieval-Augmented Generation (RAG). Built with FAISS, RocksDB, Hugging Face Transformers, and OpenAI API, with a Streamlit UI for evaluation.

## Motivation
Retrieval-Augmented Generation (RAG) is powerful but depends heavily on how documents are chunked. I built this project to explore advanced chunking strategies, compare their performance, and learn how they impact real-world RAG systems. This project helped me deepen my understanding of vector databases, LLM integration, and efficient retrieval at scale.

## Quick Start
```bash
git clone https://github.com/namanx20/Chunking-Policies-in-RAG.git
cd Chunking-Policies-in-RAG
pip install -r requirements.txt
streamlit run main.py
```

**Step 1: Clone this repository**

```bash
git clone https://github.com/namanx20/Chunking-Policies-in-RAG.git
cd Chunking-Policies-in-RAG
```

**Step 2: Set up the environment**

Create a virtual environment and install the dependencies:
```bash
conda create -n rag_system python=3.10
conda activate rag_system
pip install -r requirements.txt
```

**Step 3: Add your data**

Add a `data` folder within the repo and place the `.txt` files you want to ingest into RocksDB.

**Step 4: Ingest data into the vector database**

Run the following command to ingest your data:

```bash
python build_vector_store.py
```

**Step 5: Login to Hugging Face**

Run:

```bash
huggingface-cli login
```

When prompted, enter your Hugging Face token. Do not share your token with anyone.

**Step 6: Run the Streamlit UI**

Once the data is ingested, run:

```bash
streamlit run main.py
```

This will launch the Streamlit UI where you can upload documents and compare chunking strategies.

## Demo
![UI Screenshot](demo.png)

## Project Overview
For detailed background knowledge, methodology, and task breakdowns, see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).
