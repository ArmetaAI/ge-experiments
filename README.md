# Vector Search Setup

## Prerequisites
- Python 3.12
- Google Cloud SDK
- Cloud SQL Proxy

## Initial Setup

### 1. Create virtual environment
```bash
cd shared
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Authenticate with Google Cloud
```bash
gcloud auth application-default login
```

---

## Data Pipeline: Google Drive → Database

### Step 1: Download PDFs from Google Drive
```bash
python shared/download_google_drive.py
```
- Downloads PDFs from subfolders (ignores root)
- Renames files as `subfolder_X.pdf`
- Output: `downloaded_pdfs/` directory

### Step 2: Extract Tags from PDFs
```bash
python extract_tags_from_pdfs.py --input-dir downloaded_pdfs
```
- Processes first 2 pages with OCR
- Extracts title and keywords using Vertex AI
- Output: `results/document_tags_TIMESTAMP.csv`

Optional parameters:
- `--max-files N` - Process only first N files
- `--output path.csv` - Custom output path

### Step 3: Start Cloud SQL Proxy
Open new terminal:
```bash
cloud-sql-proxy gosexpert:europe-west1:postgres --port 5433
```

### Step 4: Import Tags to Database
```bash
python shared/import_tags_from_csv.py --input results/document_tags_TIMESTAMP.csv
```
- Generates embeddings for each tag
- Inserts into `document_tags` table
- Use `--include-errors` to import rows with extraction errors

---

## Vector Search

### Run Search
```bash
python shared/test_vector_search.py "search query" --top-k 5 --threshold 0.3
```

### Examples
```bash
python shared/test_vector_search.py "архитектурно-планировочное задание"
python shared/test_vector_search.py "строительные чертежи" --top-k 10
python shared/test_vector_search.py "проектная документация" --threshold 0.5
```

### Parameters
- `query` - Search query text (required)
- `--top-k` - Number of results (default: 5)
- `--threshold` - Minimum similarity 0-1 (default: 0.3)
