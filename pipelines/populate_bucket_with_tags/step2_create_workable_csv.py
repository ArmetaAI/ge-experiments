"""
Simple CSV builder: scan downloaded_pdfs and extract tag names from filenames
Pattern: {tag_name}_{number}.pdf
"""
import csv
from pathlib import Path
import re

DOWNLOAD_DIR = Path('downloaded_pdfs')
OUTPUT_CSV = Path('download_status_clean.csv')

print(f"Scanning {DOWNLOAD_DIR}...")

if not DOWNLOAD_DIR.exists():
    print(f"Error: {DOWNLOAD_DIR} doesn't exist!")
    exit(1)

# Pattern: tag_1.pdf, tag_2.pdf, etc.
pattern = re.compile(r'^(.+?)_(\d+)\.pdf$')

records = []
local_files = sorted(DOWNLOAD_DIR.glob('*.pdf'))

print(f"Found {len(local_files)} PDF files")

for pdf_path in local_files:
    filename = pdf_path.name
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)

    match = pattern.match(filename)
    if match:
        tag_short = match.group(1)
        file_number = match.group(2)

        records.append({
            'tag_short_name': tag_short,
            'filename': filename,
            'file_number': int(file_number),
            'file_size_mb': round(file_size_mb, 2)
        })
    else:
        print(f"  Warning: Couldn't parse filename: {filename}")

print(f"\nParsed {len(records)} files")

# Write CSV
print(f"Writing to {OUTPUT_CSV}...")
with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['tag_short_name', 'filename', 'file_number', 'file_size_mb']
    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(records)

print(f"✓ Wrote {len(records)} records to {OUTPUT_CSV}")

# Summary
from collections import defaultdict
by_tag = defaultdict(list)
for rec in records:
    by_tag[rec['tag_short_name']].append(rec)

print(f"\n=== Summary ===")
print(f"Total files: {len(records)}")
print(f"Unique tags: {len(by_tag)}")
print(f"Total size: {sum(r['file_size_mb'] for r in records):.2f} MB")

print(f"\nTop 10 tags by file count:")
sorted_tags = sorted(by_tag.items(), key=lambda x: len(x[1]), reverse=True)[:10]
for tag, files in sorted_tags:
    total_size = sum(f['file_size_mb'] for f in files)
    print(f"  {tag}: {len(files)} files, {total_size:.2f} MB")
