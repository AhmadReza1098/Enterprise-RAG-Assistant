# 🏦 Enterprise Regulatory Intelligence System

### Multi-Document RAG Pipeline & Policy QA Assistant with Chain-of-Thought Reasoning

---

## 📌 Table of Contents
* [Project Overview](#-project-overview)
* [Business & Governance Objectives](#-business--governance-objectives)
* [Data Sources & Ingestion](#-data-sources--ingestion)
* [System Architecture & Pipeline](#-system-architecture--pipeline)
* [Core Technologies & Models](#-core-technologies--models)
* [Advanced Prompt Engineering](#-advanced-prompt-engineering)
* [Evaluation Strategy & Metrics](#-evaluation-strategy--metrics)
* [Interactive Application](#-interactive-application)
* [Project Structure](#-project-structure)
* [How to Run This Project](#-how-to-run-this-project)
* [Future Roadmap](#-future-roadmap)
* [Author & Contact](#-author--contact)

---

## 🚀 Project Overview

The **Enterprise Regulatory Intelligence System** is an end-to-end Retrieval-Augmented Generation (RAG) system engineered to solve the challenge of manual regulatory compliance and dense policy navigation within financial institutions. 

By ingesting unstructured regulatory circulars (such as **Reserve Bank of India (RBI) MSME guidelines**) alongside technical knowledge bases, the system allows compliance officers, risk analysts, and internal auditors to query dense policy documents in real-time. It enforces strict **Chain-of-Thought (CoT)** reasoning to eliminate hallucinations and produce context-grounded, audit-ready answers.

---

## 🎯 Business & Governance Objectives

### 1. Accelerated Compliance & Policy Navigation
* **Objective:** Enable instant semantic search across multi-page circulars, lending norms, and regulatory mandates.
* **Why it matters:**
  * Regulatory circulars (like RBI lending frameworks) span dozens of pages of dense legalese.
  * Manual review bottlenecks credit decisions, audit workflows, and branch-level policy compliance.
  * Rapid contextual retrieval cuts policy lookup time from hours to seconds.

### 2. Elimination of AI Hallucinations in High-Stakes Finance
* **Objective:** Guarantee that all generated answers are strictly derived from verified source text.
* **Why it matters:**
  * Standard Generative AI models frequently generate confident but incorrect facts (hallucinations), posing severe financial and legal risks in banking.
  * Enforcing explicit step-by-step reasoning ensures the AI verifies each clause before giving a final ruling.

### 3. Scalable Multi-Domain Knowledge Base
* **Objective:** Dynamically index diverse document types (regulatory mandates, technical manuals, internal SOPs) without code alterations.
* **Why it matters:**
  * Financial institutions manage constantly evolving guidelines across lending, AML/KYC, cyber governance, and operations.
  * A scalable ingestion pipeline automatically updates the vector store as new circulars are published.

---

## 🗄️ Data Sources & Ingestion

The ingestion engine processes unstructured PDF documents placed within the `data/raw_documents/` repository:
* **`rbi_msme_2026.pdf`**: Reserve Bank of India regulatory framework covering Micro, Small, and Medium Enterprises (collateral-free lending thresholds, credit guarantee schemes, priority sector classifications).
* **`ai_test.pdf`**: Multi-domain technical documentation used to stress-test the engine's ability to isolate disparate subject matters within a unified index.

**Extraction Logic:**
* Documents are loaded and parsed into clean raw text strings using `pdfplumber`.
* Dynamic folder iteration automatically ingests any newly added PDF without changing application parameters.

---

## 🔍 System Architecture & Pipeline

```text
[ Unstructured PDFs ]
        │
        ▼  (pdfplumber)
[ Raw Text Extraction ]
        │
        ▼  (RecursiveCharacterTextSplitter: 1000 chunk size / 200 overlap)
[ Overlapping Semantic Chunks ]
        │
        ▼  (HuggingFace all-MiniLM-L6-v2 Bi-Encoder)
[ Dense Embedding Vectors (384-dim) ]
        │
        ▼
[ FAISS Vector Index (data/vector_db) ]
        │
        ├── User Query ──► [ Dense Query Vector ]
        │                           │
        ▼                           ▼
[ Top-K Semantic Similarity Search (K = 6) ]
        │
        ▼
[ Chain-of-Thought Prompt + Retrieved Context ]
        │
        ▼  (Groq API / Meta Llama-3.1-8B-Instant)
[ Step-by-Step Reasoning & Audit-Ready Answer ]
```

---

## ⚙️ Core Technologies & Models

* **Text Ingestion & Parsing:** `pdfplumber` for robust multi-page PDF text extraction.
* **Chunking Engine:** `langchain.text_splitter.RecursiveCharacterTextSplitter` configured with a chunk size of 1,000 characters and a 200-character sliding overlap to preserve boundary context across paragraphs.
* **Bi-Encoder Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace, generating dense 384-dimensional vector representations optimized for semantic similarity.
* **Vector Index:** `FAISS` (Facebook AI Similarity Search) utilizing L2 distance indexing for sub-millisecond nearest-neighbor retrieval.
* **LLM Generation Engine:** `Meta Llama-3.1-8B-Instant` deployed over the high-throughput `Groq` inference engine for rapid response generation.
* **Pipeline Orchestration:** `LangChain Classic` retrieval and document-stuffing chains.
* **User Interface:** `Streamlit` with dynamic session-state chat management.

---

## 🧠 Advanced Prompt Engineering

To ensure compliance-grade answers, the system bypasses naive zero-shot generation in favor of a structured **Chain-of-Thought (CoT)** prompt architecture:

```text
You are an expert financial and technical assistant.
Your goal is to answer the user's question based ONLY on the provided context.

INSTRUCTIONS:
1. First, think step-by-step about how the context relates to the question.
2. Write out your thinking process clearly.
3. Finally, provide your official answer.
4. If the answer is not in the context, say 'I cannot find this in the documents.'

Context: {context}
Question: {input}
Answer:
```

### Why this matters:
* **Explainability & Auditability:** The model provides a visible breakdown of which clauses it referenced before delivering its conclusion.
* **Empirical Hallucination Reduction:** Step-by-step reasoning constraints ground the model in retrieved chunks, preventing erroneous loan limits or fabricated policy numbers.

---

## 📊 Evaluation Strategy & Metrics

The pipeline is structured around the primary pillars of RAG evaluation:

| Dimension | Metric | Purpose |
| :--- | :--- | :--- |
| **Retriever Quality** | **Recall@K** | Measures whether the ground-truth regulatory clause was captured in the top K=6 retrieved chunks. |
| **Retriever Precision** | **Precision@K** | Measures the proportion of retrieved chunks that are genuinely relevant to the query. |
| **Context Faithfulness** | **Hallucination Rate** | Quantifies whether the generated statements are 100% mathematically and textually supported by the context. |
| **Answer Relevance** | **Query Alignment Score** | Measures how directly the generated response addresses the user's specific operational question. |

---

## 🌐 Interactive Application

The system features a **Streamlit** conversational web application with full chat history memory:

* **Contextual Reasoning Display:** Exposes the model's internal step-by-step derivation for compliance audit verification.
* **Multi-Domain Intelligence:** Seamlessly switches between querying banking loan caps (e.g., MSE collateral limits) and cross-domain documentation without context contamination.

---

## 📁 Project Structure

```text
rbi_rag_project/
├── data/
│   ├── raw_documents/
│   │   ├── rbi_msme_2026.pdf         # RBI Regulatory Master Circular
│   │   └── ai_test.pdf               # Knowledge Base Document
│   └── vector_db/
│       ├── index.faiss               # FAISS Dense Vector Index
│       └── index.pkl                 # Document Chunk Metadata Store
├── src/
│   ├── 1_ingest.py                   # Single-document text extraction
│   ├── 2_chunking.py                 # Text chunking and sliding window logic
│   ├── 3_vector_db.py                # Multi-document FAISS indexing pipeline
│   ├── 4_retrieve.py                 # Standalone semantic similarity retriever
│   ├── 5_generate.py                 # Zero-shot generation testing script
│   ├── 6_app.py                      # Basic Streamlit UI implementation
│   └── 7_chat_app.py                 # Production Chat UI with CoT Prompting
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 💻 How to Run This Project

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/Enterprise-RAG-Assistant.git](https://github.com/your-username/Enterprise-RAG-Assistant.git)
cd Enterprise-RAG-Assistant
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API Credentials
Export your Groq API key in your environment:
```bash
# Windows PowerShell
$env:GROQ_API_KEY="your_groq_api_key_here"

# Linux / macOS
export GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Build / Update the Vector Database
Place your PDF files inside `data/raw_documents/` and run the vectorization pipeline:
```bash
python src/3_vector_db.py
```

### 5. Launch the Streamlit Chat Application
```bash
streamlit run src/7_chat_app.py
```

---

## 📈 Future Roadmap

* **Cross-Encoder Re-Ranking:** Integrate a secondary `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranking step over the top-20 retrieved candidates to optimize retrieval precision.
* **Conversational Memory:** Implement `ConversationBufferWindowMemory` to support multi-turn conversational context across iterative queries.
* **Automated Benchmark Evaluation:** Implement continuous RAG evaluation using the **RAGAS** framework to score Context Precision, Faithfulness, and Answer Relevance against a gold-standard regulatory test set.

---

## 👤 Author & Contact

**Ahmad Reza**  
*Quantitative Economics & Data Analytics*  
* 📧 **Email:** ahmadreza6122@gmail.com
* 🔗 **LinkedIn:** [linkedin.com/in/ahmad-reza-econ](https://www.linkedin.com/in/ahmad-reza-econ)
* 🔗 **GitHub:** [github.com/your-username](https://github.com/your-username)