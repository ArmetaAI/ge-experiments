# Vector Search Setup

## Prerequisites
- Python 3.12
- Google Cloud SDK
- Cloud SQL Proxy

## Setup

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Authenticate with Google Cloud
```bash
gcloud auth application-default login
```

### 4. Start Cloud SQL Proxy
```bash
cloud-sql-proxy gosexpert:europe-west1:postgres --port 5433
```

## Run Vector Search

```bash
python test_vector_search.py "search query" --top-k 5 --threshold 0.3
```

### Examples
```bash
python test_vector_search.py "архитектурно-планировочное задание"
python test_vector_search.py "строительные чертежи" --top-k 10
python test_vector_search.py "проектная документация" --threshold 0.5
```

## Parameters
- `query` - Search query text (required)
- `--top-k` - Number of results (default: 5)
- `--threshold` - Minimum similarity 0-1 (default: 0.3)
