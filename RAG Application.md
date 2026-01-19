# 🧠 RAG — Pipeline Summary (By Code File)

This document summarizes each stage of the RAG system and explains what process is performed by each corresponding Python file.

---

## 📂 1. `ingestion_pipeline.py` — Document Preparation

**Purpose:**  
Prepare documents for retrieval by converting them into embeddings and storing them in a vector database.

**Main Processes:**
- Load raw `.txt` documents
- Split documents into smaller chunks
- Convert chunks to embeddings
- Store embeddings + metadata into Chroma DB
- Persist DB to disk for reuse

**Resulting Output:**
```
Documents → Chunks → Embeddings → Vector Store
```

---

## 📂 2. `retrieval_pipeline.py` — Context Retrieval

**Purpose:**  
Search Vector DB for document chunks related to a given user query.

**Main Processes:**
- Load persisted Chroma DB
- Embed the user query
- Perform similarity search
- Return relevant chunks as context

**Resulting Output:**
```
User Query → Relevant Chunks (Context)
```

---

## 📂 3. `answer_generation.py` — Final Answer Construction

**Purpose:**  
Generate an answer using retrieved chunks as factual context.

**Main Processes:**
- Retrieve relevant chunks
- Inject chunks into the LLM prompt
- Instruct LLM to answer using provided context only
- Apply hallucination guardrails

**Resulting Output:**
```
Context + Query → LLM Answer (Grounded)
```

---

## 📂 4. `chat_pipeline.py` — Conversational RAG with Memory

**Purpose:**  
Support multi-turn chat and handle follow-up questions.

**Main Processes:**
- Maintain conversation history
- Rewrite follow-up questions into standalone queries
- Retrieve context again for rewritten query
- Generate grounded answer
- Store Q&A into memory

**Resulting Output:**
```
Dialogue → Rewrite → Retrieve → Answer → Memory
```

---

## 📂 5. `chunking_methods.py` — Character-Based Chunking

**Purpose:**  
Split documents by size or simple structural rules.

**Main Processes:**
- Use fixed chunk size limits
- Split using separators such as space or newline

**Output:**
```
Text → Mechanical Chunks (size-based)
```

---

## 📂 6. `semantic_chunking.py` — Meaning-Based Chunking

**Purpose:**  
Split text based on semantic similarity between sentences.

**Main Processes:**
- Embed sentences
- Measure sentence similarity
- Detect topic shifts (breakpoints)
- Group related sentences together

**Output:**
```
Text → Topic-Coherent Chunks
```

---

## 📂 7. `agentic_chunking.py` — LLM-Decided Chunking

**Purpose:**  
Allow LLM to split text at natural human topics.

**Main Processes:**
- Prompt LLM with chunking rules
- LLM inserts split markers
- Parser converts into final chunks

**Output:**
```
Text → Human-Like Chunks (LLM Reasoned)
```

---

## 📂 8. `multi_query_retrieval.py` — Multi-Query Expansion

**Purpose:**  
Increase retrieval recall by reformulating the query.

**Main Processes:**
- LLM generates multiple semantic variations of the query
- Retrieve chunks for each variation
- Collect all retrieved results

**Output:**
```
One Query → Many Reformulated Queries → Many Retrieved Results
```

---

## 📂 9. `multi_query_rrf_retrieval.py` — Fusion & Ranking (RRF)

**Purpose:**  
Fuse multiple retrieval result lists into a single better-ranked list.

**Main Processes:**
- Apply Reciprocal Rank Fusion scoring
- Boost documents appearing across multiple lists
- Sort by final fused score

**Output:**
```
Multiple Rankings → Fused Final Ranking
```

---

# 🧩 Overall RAG Flow (All Files Together)

```
(1) ingestion_pipeline.py
          ↓
(2) retrieval_pipeline.py
          ↓
(3) answer_generation.py
          ↓
(4) chat_pipeline.py (optional multi-turn)
          ↓
(5) multi_query + RRF (optional optimization)
```

---

# 🏁 One-Line Summaries

| File | Summary |
|---|---|
| ingestion_pipeline | Prepare docs for search |
| retrieval_pipeline | Find relevant chunks |
| answer_generation | Answer using retrieved data |
| chat_pipeline | Multi-turn conversational RAG |
| chunking_methods | Size-based splitting |
| semantic_chunking | Meaning-based splitting |
| agentic_chunking | LLM-driven splitting |
| multi_query_retrieval | Query reformulation for recall |
| multi_query_rrf_retrieval | Fusion for better ranking |

---
