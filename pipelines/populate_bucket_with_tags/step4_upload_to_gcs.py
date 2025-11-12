import pandas as pd
from google.cloud import storage
import os
from pathlib import Path

# Initialize Google Cloud Storage client
client = storage.Client()
bucket_name = 'gosexpert_categorize'
bucket = client.bucket(bucket_name)

# Read the CSV file
df = pd.read_csv('download_status_with_tags.csv', encoding='utf-8')

# Filter rows that have valid data
df = df.dropna(subset=['filename', 'document_tag', 'file_type'])

print(f"Total files to upload: {len(df)}")

# Base directory where PDFs are stored
pdf_base_dir = 'downloaded_pdfs'

# Upload each PDF file
for index, row in df.iterrows():
    filename = row['filename']
    document_tag = row['document_tag']
    file_type = row['file_type']

    # Construct the full path to the PDF
    pdf_path = os.path.join(pdf_base_dir, filename)

    # Check if the PDF file exists
    if not os.path.exists(pdf_path):
        print(f"⚠️  File not found: {pdf_path}")
        continue

    # Create blob name (you can customize the path structure)
    blob_name = f"{filename}"
    blob = bucket.blob(blob_name)

    # Set metadata
    metadata = {
        'tag': str(document_tag),
        'file_type': str(file_type)
    }
    blob.metadata = metadata

    # Upload the file
    try:
        blob.upload_from_filename(pdf_path)
        print(f"✓ Uploaded: {filename} (tag: {document_tag}, type: {file_type})")
    except Exception as e:
        print(f"✗ Failed to upload {filename}: {str(e)}")

print("\nUpload complete!")
