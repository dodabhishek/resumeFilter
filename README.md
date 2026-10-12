# AI Resume Screening & Ranking System

An intelligent, production-minded resume screening and ranking backend and Web UI built for high-throughput evaluation of candidates against dynamic job requirements.

---

## 📌 Overview

The system ingests candidate resumes (`.pdf`, `.docx`, `.doc`, `.txt`), deterministically enforces **dynamic user requirements**, strictly verifies that candidates have **hands-on project implementation** (rejecting candidates who merely list keywords in their skills section without building projects), and displays explainable candidate decision cards in an interactive Web UI.

---

## 🌟 Key Features

1. **Interactive Web UI**:
   - Modern, responsive single-page application served directly at `http://localhost:8000/`.
   - **Resume Ingestion**: Upload individual files or an entire folder of resumes (`.pdf`, `.docx`, `.doc`, `.txt`), or screen the built-in 51 benchmark resumes.
   - **Dynamic Job Criteria Input**:
     - Target Job Role / Title.
     - **⭐ Must Have Built Projects With**: Technologies the candidate MUST have built projects using.
     - **Core Required Skills**: Mandatory foundation skills.
     - **Preferred / Bonus Skills**: Nice-to-have technologies.
     - Quick one-click presets for common engineering profiles.
   - **Dual Tab Decision View**:
     - 🟢 **Accepted Candidates Tab**: Sorted by match fit (raw scores are hidden); displays candidate name, email, GitHub link, **What They Built** project summary, and bullet-point **Why Accepted** evaluation remarks.
     - 🔴 **Rejected Candidates Tab**: Displays candidate details and explicit bullet-point **Why Rejected** remarks (e.g. lacking project experience with required technologies, missing core skills, unreadable files).
     - Search & filter candidates in real time.
     - One-click JSON export.

2. **Hands-On Project Verification Engine**:
   - Analyzes candidate project descriptions and work experience.
   - Distinguishes between genuine project usage and keyword stuffing in a skills list.
   - Candidates who only mention required technologies in a skills list without project evidence are rejected with clear remarks.

3. **Multi-Tiered Scoring Rubric & Quality Penalties**:
   - 40 pts: Project Depth & Architecture.
   - 30 pts: Core Technical Foundations.
   - 15 pts: Cloud, Deployment & Full-Stack integration.
   - 10 pts: Public GitHub activity & maintained repositories.
   - 5 pts: Engineering hygiene (pytest, CI/CD, observability).
   - **Deductions**: -5 to -15 pts for shallow single-prompt API wrappers, -5 to -10 pts for tutorial copies.

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.10 – 3.14)
- **Zero mandatory external dependencies** required for core execution (includes pure-Python standard library fallbacks for PDF parsing, text extraction, JSON validation, and HTTP server).

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/dodabhishek/resumeFilter.git
cd resumeFilter

# Copy environment template
cp .env.example .env
```

### 3. Launch Web UI & Server
```bash
python3 server.py --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser:
- Select your target role and project requirements.
- Drag & drop resume files, select a folder, or use `./resumes`.
- Click **"Start Filtering Resumes"** to screen candidates in real-time.

### 4. Run CLI Pipeline with Dynamic Criteria
```bash
# Default screening on local resumes folder
python3 main.py --input ./resumes --output ./output/results.json

# Screen with custom dynamic criteria
python3 main.py \
  --role "AI / Agentic Engineer" \
  --project-required "LangGraph, RAG, OpenAI" \
  --required "Python, FastAPI" \
  --preferred "Docker, Redis" \
  --input ./resumes
```

### 5. Run Unit Tests
```bash
python3 -m unittest discover tests
```

---

## 📂 Project Structure

```
.
├── static/
│   └── index.html            # Modern Web UI (Uploader, Dynamic Criteria, Accepted/Rejected Tabs)
├── src/
│   ├── config.py             # Centralized settings & scoring weights
│   ├── enrichment/
│   │   └── github.py         # GitHub API client with caching & rate-limit resilience
│   ├── llm/
│   │   ├── client.py         # Dynamic LLM client adapter (Mock, OpenAI, Gemini)
│   │   └── prompts.py        # Dynamic prompt generator with role & criteria injection
│   ├── models/
│   │   └── schemas.py        # Schemas (JobCriteria, CandidateResult, BatchSummary, ParsedResume)
│   ├── parser/
│   │   ├── cleaner.py        # Regex extraction (emails, GitHub handles, sections)
│   │   └── extractor.py      # Resilient PDF/DOCX/TXT document parsing engine
│   ├── screening/
│   │   ├── filter.py         # Dynamic eligibility & hands-on project verification filter
│   │   └── scorer.py         # Dynamic scoring engine with penalty detection
│   └── pipeline.py           # ThreadPoolExecutor batch orchestrator
├── tests/
│   ├── test_dynamic_criteria.py # Unit tests for dynamic user requirements & project verification
│   ├── test_eligibility.py   # Unit tests for hard filtering edge cases
│   ├── test_parser.py        # Unit tests for text & metadata extraction
│   ├── test_scoring.py       # Unit tests for scoring rubric and penalties
│   └── test_edge_cases.py    # Edge case testing
├── resumes/                  # 51 synthetic candidate test resumes
├── output/
│   └── results.json          # Formatted screening output
├── main.py                   # CLI entrypoint with dynamic criteria arguments
├── server.py                 # Web UI server & REST API endpoints
├── requirements.txt          # Package requirements (optional enhancements)
├── .env.example              # Environment variables template
└── README.md
```

---

## 📡 REST API Endpoints

- `GET /` — Serves the interactive Single Page Web UI.
- `GET /health` — Health check status.
- `POST /api/upload` — Multipart file/folder upload supporting `.pdf`, `.docx`, `.doc`, `.txt`.
- `POST /api/screen` — Trigger screening batch with dynamic `JobCriteria` payload.
- `GET /api/results` — Retrieve latest screening results and batch summary.
