# 🧠 Retrieval Augmented Generation (RAG) - How This Application Works

#  **Ingestion pipeline**


This document explains how the RAG application works based on the provided ingestion pipeline code.  
The pipeline prepares documents for RAG usage by converting them into embeddings and storing them inside a vector database.

---

# 📍 1. High-Level RAG Pipeline

The ingestion pipeline implements the first half of a RAG workflow:

```
Documents → Chunks → Embeddings → Vector Store
```

This prepares data so later query stages can perform retrieval.

---

# 📍 2. Components of the Ingestion Code

The pipeline executes **three main stages**:

```
1. Load Documents
2. Split Documents into Chunks
3. Embed & Store in Vector DB
```

---

# 📍 3. Step-by-Step Breakdown

## **(1) Load Documents – `load_documents()`**

**Purpose:** Import raw company documents into memory.

### 🔹 Internal Operations:

- Verify that the `docs` folder exists
- Load all `.txt` files using `DirectoryLoader`
- Wrap each file into LangChain `Document` objects:

```
Document(
  page_content="Tesla is a clean energy company...",
  metadata={"source": "docs/tesla.txt"}
)
```

Metadata allows tracing retrieved chunks back to files.

---

## **(2) Split into Chunks – `split_documents()`**

**Purpose:** Break large documents into manageable text chunks.

LLMs and embeddings cannot operate efficiently on very long text, so documents are split into smaller pieces.

### 🔹 Internal Behavior:

Given:
```
chunk_size = 1000
chunk_overlap = 0
```

A 3000-character document becomes:

```
chunk 1: 0-999
chunk 2: 1000-1999
chunk 3: 2000-2999
```

Each chunk still carries metadata such as its source file.

---

## **(3) Embedding & Storage – `create_vector_store()`**

This is the core RAG ingestion step.

### **Step A — Compute Embeddings**

The model:
```
OpenAIEmbeddings(model="text-embedding-3-small")
```

Converts chunks into numerical vectors (embeddings) representing semantic meaning, e.g.:

```
[0.12, 0.88, 0.02, ...]
```

### **Step B — Store in Vector Database (Chroma)**

Chroma stores:

| Item | Purpose |
|---|---|
| Embedding | Semantic search |
| Chunk text | Context for LLM |
| Metadata | Traceability |
| ID | Document indexing |

Storage is persisted to disk under:

```
db/chroma_db/
```

### **Step C — Persistence**

Embedding data is saved so ingestion runs only once.

---

# 📍 4. Vectorstore Reuse on Next Runs

Before new ingestion, code checks:

```
if os.path.exists("db/chroma_db"):
```

If exists:

✔ Skip reprocessing  
✔ Load vector store  
✔ Ready for semantic retrieval immediately

---

# 📍 5. What This Pipeline Enables

After ingestion, the system can perform:

```
User Query → Vector Search → Relevant Chunks → LLM Answer
```

Example for query:

> "What does Tesla do?"

The RAG engine retrieves relevant chunks instead of letting LLM hallucinate.

---

# 📍 6. What is Not Covered Yet (Handled in Later Stages)

This ingestion code does **not**:

❌ Generate answers  
❌ Retrieve chunks for questions  
❌ Handle chat history  
❌ Perform reranking or hybrid search  

These are handled by:

- `2_retrieval_pipeline.py`
- `3_answer_generation.py`
- `4_history_aware_generation.py`

---

# 📍 7. Final Summary

The ingestion pipeline performs:

```
Raw Documents
      ↓
Loading (metadata)
      ↓
Chunking (segmentation)
      ↓
Embedding (semantic vectors)
      ↓
Vector Store Persistence (Chroma)
```

This prepares all required data for RAG-based semantic Q&A, enterprise search, and chatbot applications.

---

#  **Retrieval pipeline**


This document explains the retrieval stage of a RAG (Retrieval Augmented Generation) system based on the provided code.

The retrieval pipeline is responsible for finding the most relevant information from stored documents before an LLM generates the answer.

---

# 📍 1. Purpose of Retrieval Stage

Once documents have been ingested into a Vector Database, the retrieval stage does:

```
User Query → Vector Search → Relevant Context
```

Retrieval ensures that responses are based on real knowledge from documents instead of LLM hallucinations.

---

# 📍 2. Code Overview

Key components used in the code:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
```

Retrieval pipeline steps:

1. Load VectorDB (Chroma)
2. Embed the user query
3. Perform similarity search
4. Return the top relevant document chunks

---

# 📍 3. Step-by-Step Breakdown

## **(1) Load Vector Store + Embeddings**

```python
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)
```

### 🧠 Internal Behavior:
- Loads stored document embeddings from disk
- Loads HNSW similarity index
- Sets cosine similarity as search metric

This allows the system to search documents efficiently.

---

## **(2) Prepare Query**

```python
query = "How much did Microsoft pay to acquire GitHub?"
```

This is the natural language question from the user.

---

## **(3) Execute Similarity Search**

```python
retriever = db.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke(query)
```

This step finds the most similar chunks.

### 🧠 Internal Breakdown:

#### **Step A — Query Embedding**
The query is converted into a vector:

```
[0.12, 0.56, 0.88, ...]
```

#### **Step B — Compare Against Stored Embeddings**
Chroma computes cosine similarity between:

```
Query Vector ↔ Document Chunk Vectors
```

#### **Step C — Rank & Select**
Returns top `k=5` chunks with highest similarity scores.

This is the **retrieval** part of RAG.

---

## **Optional Threshold Filtering**

```python
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3}
)
```

This filters out irrelevant chunks to reduce hallucination.

---

## **(4) Display Retrieved Context**

```python
for doc in relevant_docs:
    print(doc.page_content)
```

This prints the text content of retrieved chunks.  
These chunks are called **context**.

---

# 📍 4. Why Retrieval Matters in RAG

Retrieval ensures:

✔ factual answers  
✔ evidence-based responses  
✔ prevents hallucination  
✔ supports private knowledge

Without retrieval:

LLM guesses and may answer incorrectly.

With retrieval:

LLM uses verified context from documents.

---

# 📍 5. Retrieval in Full RAG Workflow

Retrieval sits between ingestion and generation:

```
(1) Ingestion
      ↓
(2) Retrieval ← (this code)
      ↓
(3) Generation (LLM final answer)
```

Retrieval provides the LLM with real context for answering questions.

---

# 📍 6. Final Summary (Simple)

➡ **Execute Similarity Search** = find document chunks related to query  
➡ **Display Context** = show those chunks for LLM usage

Together:

```
User Query → Retrieved Context → LLM → Final Answer
```

This completes the retrieval stage of RAG.

---
