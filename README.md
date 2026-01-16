# 📊 Financial Document RAG Assistant with Reasoning

A sophisticated RAG (Retrieval-Augmented Generation) system powered by NVIDIA's Nemotron 70B reasoning model for advanced financial document analysis with hybrid search, table extraction, and chart OCR capabilities.

## 🌟 Features

- **🧠 Advanced Reasoning Model**: NVIDIA Nemotron 70B for step-by-step analytical reasoning
- **🔍 Hybrid Search**: Combines vector embeddings (ChromaDB) with BM25 keyword search
- **📊 Table Extraction**: Automatically extracts tables from PDFs using Camelot and Tabula
- **📈 Chart OCR**: Uses Tesseract OCR to extract text from charts and images
- **🎯 Zero Hallucination**: Strict grounding to source documents only
- **📝 Source Citations**: Every answer includes page and chunk references
- **💡 Step-by-Step Analysis**: Provides detailed reasoning for complex questions
- **💬 Interactive Chat Interface**: Streamlit-based UI with conversation history

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Example Questions](#example-questions)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)
- [Contributing](#contributing)
- [License](#license)

---

## 🔧 Prerequisites

### System Dependencies

The application requires several system-level tools for PDF processing, OCR, and table extraction.

#### **Ubuntu/Debian Linux:**
```bash
sudo apt-get update
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    ghostscript \
    poppler-utils \
    python3-opencv \
    libgl1-mesa-glx
```

#### **macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install tesseract
brew install poppler
brew install ghostscript
```

#### **Windows:**

1. **Tesseract OCR:**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to: `C:\Program Files\Tesseract-OCR`
   - Add to PATH: `C:\Program Files\Tesseract-OCR`

2. **Poppler:**
   - Download from: http://blog.alivate.com.au/poppler-windows/
   - Extract to: `C:\poppler`
   - Add to PATH: `C:\poppler\Library\bin`

3. **Ghostscript:**
   - Download from: https://www.ghostscript.com/download/gsdnld.html
   - Install and add to PATH

**Verify installations:**
```bash
tesseract --version
gs --version
pdfinfo -v
```

---

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
# Option 1: If using git
git clone <your-repo-url>
cd financial-rag-assistant

# Option 2: Download and extract the zip file
# Then navigate to the extracted folder
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** Installation may take 5-10 minutes due to large dependencies (PyTorch, etc.)

### Step 4: Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your favorite editor
nano .env    # or vim, code, notepad, etc.
```

### Step 5: Get Your NVIDIA API Key

1. Go to https://build.nvidia.com/
2. Sign up or log in with your account
3. Navigate to "API Catalog"
4. Click "Get API Key" or "Generate Key"
5. Copy your API key (starts with `nvapi-`)

### Step 6: Add API Key to .env

Edit your `.env` file:
```bash
NVIDIA_API_KEY=nvapi-your-actual-key-here
```

**Important:** Never commit the `.env` file to version control!

---

## ⚙️ Configuration

### Default Settings

The application comes pre-configured with optimal settings in `app.py`:

```python
# Model Configuration
REASONING_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"  # Reasoning model
EMBED_MODEL = "nvidia/nv-embed-v1"                           # Embedding model

# Search Configuration
DIST_THRESHOLD = 1.2  # Distance threshold for relevance
k = 5                 # Number of chunks to retrieve

# Text Splitting
chunk_size = 900      # Characters per chunk
chunk_overlap = 150   # Overlap between chunks
```

### Available NVIDIA Models

You can change the reasoning model in `app.py`:

**Reasoning Models (Recommended for Analysis):**
- `nvidia/llama-3.1-nemotron-70b-instruct` - **Best for reasoning** (default)
- `meta/llama3-70b-instruct` - Good balance of speed and quality
- `mistralai/mixtral-8x7b-instruct-v0.1` - Alternative reasoning model

**Faster Models (For Simple Queries):**
- `meta/llama3-8b-instruct` - Faster, lighter model
- `mistralai/mistral-7b-instruct-v0.2` - Quick responses

**Embedding Models:**
- `nvidia/nv-embed-v1` - Best quality (default)
- `nvidia/nv-embedqa-e5-v5` - Alternative embedding model

---

## 🎮 Usage

### Starting the Application

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Start the Streamlit app
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Step-by-Step Usage Guide

#### 1. **Upload a PDF Document**
   - Click "Browse files" or drag & drop a PDF
   - Supported: Financial reports, research papers, invoices, contracts
   - The system will automatically:
     - Extract text from all pages
     - Identify and extract tables
     - Perform OCR on charts and images
     - Create searchable chunks

#### 2. **Wait for Indexing**
   - Progress shown with spinner
   - Success message shows number of chunks indexed
   - Example: "Indexed 247 chunks (text + tables + charts) ✅"

#### 3. **Ask Questions**
   - Type your question in the chat input
   - Press Enter or click Send
   - The AI will:
     - Search for relevant content
     - Reason through the answer step-by-step
     - Provide citations to source pages

#### 4. **View Results**
   - Main answer with reasoning
   - Source citations at the bottom
   - Click "🔍 Retrieved Evidence" to see:
     - Exact text chunks used
     - Relevance scores
     - Page numbers

#### 5. **Continue Conversation**
   - Ask follow-up questions
   - Request clarifications
   - Compare different sections
   - Chat history is maintained

---

## 💡 Example Questions

### Financial Analysis (Leverages Reasoning Capabilities)

**Revenue & Growth:**
```
- What was the total revenue and how did it compare to last year?
- Analyze the revenue growth rate and explain the key drivers
- Why did revenue increase in Q3 but decrease in Q4?
- Compare revenue performance across different product lines
```

**Profitability:**
```
- Analyze the profit margins and explain any significant changes
- Why did the gross margin change between quarters?
- What factors contributed to the improvement in operating margin?
- Calculate and explain the EBITDA margin trend
```

**Expenses:**
```
- What were the operating expenses by category?
- Explain the increase in R&D spending
- How do SG&A expenses compare to industry standards?
- Analyze the cost structure and identify optimization opportunities
```

### Data Extraction

**Financial Statements:**
```
- Show me the income statement breakdown
- Extract the balance sheet key figures
- List all assets and liabilities
- What are the cash flow components?
```

**Specific Metrics:**
```
- What is the debt-to-equity ratio?
- List all the financial metrics mentioned
- Show me the quarterly earnings per share
- What are the liquidity ratios?
```

### Comparative Analysis

**Performance Comparison:**
```
- Compare the performance across different business segments
- How do current ratios compare to the previous year?
- Analyze regional performance differences
- Compare this quarter's performance to the same quarter last year
```

**Trend Analysis:**
```
- What trends do you see in the revenue data?
- Analyze the expense trends over the past year
- How has the company's profitability evolved?
- Identify any concerning patterns in the financial data
```

### Risk & Strategy

**Risk Assessment:**
```
- What are the key financial risks mentioned?
- Identify liquidity concerns from the financial statements
- What operational risks are highlighted?
- Analyze the debt repayment schedule and risks
```

**Strategic Insights:**
```
- What strategic initiatives are mentioned?
- Summarize the management discussion and analysis
- What are the growth opportunities identified?
- Explain the capital allocation strategy
```

---

## 🔬 How It Works

### Architecture Overview

```
┌─────────────┐
│   PDF File  │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────┐
│   Document Processing Layer      │
│  ┌────────┬─────────┬─────────┐ │
│  │  Text  │ Tables  │  Charts │ │
│  │Extract │ Extract │   OCR   │ │
│  └────────┴─────────┴─────────┘ │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│      Chunking & Indexing         │
│  ┌──────────────────────────┐   │
│  │  Text Splitter (900ch)   │   │
│  │  Overlap: 150 chars      │   │
│  └──────────────────────────┘   │
└──────────────┬───────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌──────────────┐
│  ChromaDB   │  │  BM25 Index  │
│  (Vector)   │  │  (Keyword)   │
└─────────────┘  └──────────────┘
       │                │
       └───────┬────────┘
               ▼
    ┌──────────────────┐
    │  Hybrid Search   │
    │  (Top-k=5)       │
    └─────────┬────────┘
              │
              ▼
    ┌──────────────────┐
    │  NVIDIA Nemotron │
    │  70B Reasoning   │
    └─────────┬────────┘
              │
              ▼
    ┌──────────────────┐
    │  Answer + Cites  │
    └──────────────────┘
```

### Detailed Process Flow

#### 1. **Document Ingestion**

**Text Extraction:**
- Reads PDF using PyPDF
- Extracts text from each page
- Preserves page numbers for citation

**Table Extraction:**
- Primary: Camelot (stream flavor) for structured tables
- Fallback: Tabula for complex table layouts
- Converts tables to text format while preserving structure

**Chart OCR:**
- Converts PDF pages to images (300 DPI)
- Applies grayscale conversion
- Uses Tesseract OCR to extract text from images
- Filters for pages containing numerical data

#### 2. **Text Chunking**

```python
RecursiveCharacterTextSplitter(
    chunk_size=900,      # Max characters per chunk
    chunk_overlap=150    # Overlap for context preservation
)
```

**Strategy:**
- Splits on paragraphs first
- Then sentences
- Finally characters
- Maintains semantic coherence
- Preserves context with overlap

#### 3. **Dual Indexing**

**Vector Index (ChromaDB):**
- Generates embeddings using NVIDIA nv-embed-v1
- 1024-dimensional vectors
- Enables semantic similarity search
- Fast approximate nearest neighbor search

**Keyword Index (BM25):**
- Tokenizes text into words
- Builds inverted index
- Enables exact keyword matching
- Complements semantic search

#### 4. **Hybrid Search**

```python
def hybrid_search(query, k=5):
    # 1. Vector search
    semantic_results = chromadb.query(query, k=5)
    
    # 2. BM25 keyword search
    keyword_results = bm25.get_scores(tokenize(query))
    
    # 3. Merge and re-rank
    combined = merge_by_score(semantic_results, keyword_results)
    
    return top_k(combined, k=5)
```

**Scoring:**
- Vector similarity: `1 / (1 + distance)`
- BM25 score: Normalized TF-IDF with document length adjustment
- Final score: Sum of both scores

#### 5. **Answer Generation with Reasoning**

**Enhanced Prompt Structure:**
```
You are a financial document analyst.

CRITICAL RULES:
1. Use ONLY provided context
2. If not found, say "Not found in the document."
3. For numbers, cite exact figures
4. Explain reasoning step-by-step

Context: [Retrieved chunks with page numbers]

Question: [User query]

Think through this step-by-step:
1. What information does context provide?
2. Does it directly answer the question?
3. What's the complete answer?

Answer:
```

**Model Processing:**
- NVIDIA Nemotron 70B analyzes context
- Constructs reasoning chain
- Validates against source material
- Generates grounded response
- Adds source citations

---

## 📁 Project Structure

```
financial-rag-assistant/
│
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── .env.example           # Template for .env
├── README.md              # This file
│
├── venv/                  # Virtual environment (created during setup)
│   └── ...
│
└── .streamlit/            # Streamlit config (optional)
    └── config.toml
```

### File Descriptions

**app.py:**
- Main application logic
- PDF processing functions
- Search and indexing
- UI components
- ~400 lines of code

**requirements.txt:**
- All Python package dependencies
- Pinned versions for reproducibility
- Includes ML models, PDF tools, OCR libraries

**.env:**
- Stores sensitive configuration
- NVIDIA API key
- Never commit to git!

**README.md:**
- Documentation you're reading
- Setup instructions
- Usage examples

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. **Tesseract Not Found**

**Error:**
```
TesseractNotFoundError: tesseract is not installed
```

**Solution:**
```bash
# Verify installation
tesseract --version

# If not found, install:
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Add to PATH
# C:\Program Files\Tesseract-OCR
```

#### 2. **Poppler Not Found**

**Error:**
```
PDFInfoNotInstalledError: Unable to get page count
```

**Solution:**
```bash
# Ubuntu/Debian:
sudo apt-get install poppler-utils

# macOS:
brew install poppler

# Windows: Add to PATH
# C:\poppler\Library\bin
```

#### 3. **NVIDIA API Key Invalid**

**Error:**
```
❌ NVIDIA_API_KEY not set (check .env)
or
AuthenticationError: Invalid API key
```

**Solution:**
1. Check `.env` file exists in project root
2. Verify key format: `nvapi-xxxxx...`
3. Regenerate key at https://build.nvidia.com/
4. Restart the application

#### 4. **Out of Memory**

**Error:**
```
RuntimeError: CUDA out of memory
or
MemoryError
```

**Solution:**
```python
# In app.py, reduce chunk size:
chunk_size = 500  # Default: 900

# Reduce number of results:
k = 3  # Default: 5

# Process smaller PDFs
# Split large documents into sections
```

#### 5. **Ghostscript Error**

**Error:**
```
CalledProcessError: Ghostscript not found
```

**Solution:**
```bash
# Ubuntu/Debian:
sudo apt-get install ghostscript

# macOS:
brew install ghostscript

# Windows:
# Install from https://www.ghostscript.com/
```

#### 6. **Table Extraction Fails**

**Symptom:**
Tables not detected or extracted incorrectly

**Solution:**
```python
# Try different extraction methods:
# 1. Camelot with different flavor
tables = camelot.read_pdf(pdf, flavor="lattice")  # Instead of "stream"

# 2. Adjust Tabula settings
dfs = tabula.read_pdf(pdf, lattice=True, multiple_tables=True)

# 3. Pre-process PDF
# Ensure PDF is not scanned image
# Use OCR first if needed
```

#### 7. **Slow Performance**

**Symptoms:**
- Slow indexing
- Long wait times for answers

**Solutions:**
```python
# 1. Reduce OCR resolution
pages = convert_from_path(pdf, dpi=150)  # Instead of 300

# 2. Disable chart OCR for text-only docs
# Comment out OCR section in ingest_pdf_once()

# 3. Use faster model
REASONING_MODEL = "meta/llama3-8b-instruct"

# 4. Reduce search results
k = 3  # Instead of 5
```

#### 8. **Import Errors**

**Error:**
```
ModuleNotFoundError: No module named 'xyz'
```

**Solution:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Or install specific package
pip install package-name
```

#### 9. **Streamlit Port Already in Use**

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process
# Linux/macOS:
lsof -ti:8501 | xargs kill -9

# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

#### 10. **PDF Permission Errors**

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Check file permissions
chmod 644 your_file.pdf

# Or use sudo (Linux/macOS)
sudo streamlit run app.py

# Windows: Run as Administrator
```

---

## 🔧 Advanced Configuration

### Custom Prompts

Edit the prompt in `answer_question()` function:

```python
prompt = f"""You are a [YOUR CUSTOM ROLE].

CUSTOM RULES:
1. [Your rule 1]
2. [Your rule 2]

Context:
{context}

Question: {q}

Answer:"""
```

### Adjusting Search Parameters

```python
# In hybrid_search() function:

# Retrieve more chunks for complex queries
def hybrid_search(query, k=10):  # Increased from 5

# Adjust scoring weights
semantic_weight = 0.7
keyword_weight = 0.3
score = (semantic_score * semantic_weight) + (keyword_score * keyword_weight)
```

### Custom Chunk Sizes

```python
# In split_text() function:
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,      # Larger chunks
    chunk_overlap=200,    # More overlap
    separators=["\n\n", "\n", ". ", " "]  # Custom separators
)
```

### Adding More Models

```python
# Add in CONFIG section:
AVAILABLE_MODELS = {
    "reasoning": "nvidia/llama-3.1-nemotron-70b-instruct",
    "fast": "meta/llama3-8b-instruct",
    "balanced": "meta/llama3-70b-instruct"
}

# Add UI selector in sidebar:
selected_model = st.selectbox("Choose Model", AVAILABLE_MODELS.keys())
llm = ChatNVIDIA(model=AVAILABLE_MODELS[selected_model])
```

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Reporting Bugs

1. Check existing issues first
2. Create detailed bug report with:
   - Python version
   - OS and version
   - Error messages
   - Steps to reproduce

### Feature Requests

1. Open an issue with `[Feature Request]` tag
2. Describe the feature
3. Explain use cases
4. Provide examples if possible

### Code Contributions

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit: `git commit -m "Add feature: description"`
5. Push: `git push origin feature-name`
6. Open Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add comments for complex logic
- Update documentation
- Add type hints where applicable

---

