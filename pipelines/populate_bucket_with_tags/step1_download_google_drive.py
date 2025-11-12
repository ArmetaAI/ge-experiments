import os
import pickle
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict
import csv
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.pickle.
# Using broader drive scope to access shared folders
SCOPES = ['https://www.googleapis.com/auth/drive']


def authenticate_google_drive(credentials_file: str = 'client_secret_833344150609-9b58bca2au71apgs3uiv0dbkif0oobgn.apps.googleusercontent.com.json'):
    """
    Authenticate with Google Drive API using OAuth credentials.

    Args:
        credentials_file: Path to the OAuth client secret JSON file

    Returns:
        Authenticated Google Drive service
    """
    creds = None
    token_path = 'token.pickle'

    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    return service


def get_folder_name_by_id(service, folder_id: str) -> str:
    """
    Get folder name by folder ID.

    Args:
        service: Google Drive API service
        folder_id: The folder ID

    Returns:
        Folder name
    """
    try:
        folder = service.files().get(
            fileId=folder_id,
            fields='name',
            supportsAllDrives=True
        ).execute()
        return folder.get('name', 'unknown')
    except Exception as e:
        logger.error(f"Failed to get folder name for {folder_id}: {e}")
        logger.error("Make sure the folder is shared with your Google account!")
        return 'unknown'


def extract_folder_id_from_url(url: str) -> str:
    """
    Extract folder ID from Google Drive URL.

    Args:
        url: Google Drive folder URL

    Returns:
        Folder ID
    """
    # URL format: https://drive.google.com/drive/folders/FOLDER_ID
    if '/folders/' in url:
        return url.split('/folders/')[-1].split('?')[0]
    return url


def list_all_files_recursively(service, folder_id: str, current_folder_name: Optional[str] = None) -> List[Dict]:
    """
    Recursively list all files from a Google Drive folder.

    Args:
        service: Google Drive API service
        folder_id: The folder ID to start from
        current_folder_name: Name of the current folder (immediate parent)

    Returns:
        List of file metadata dictionaries with 'id', 'name', 'folder_name', and 'mimeType'
    """
    files_list = []
    page_token = None

    # Get current folder name
    if current_folder_name is None:
        current_folder_name = get_folder_name_by_id(service, folder_id)
        logger.info(f"Starting from root folder: {current_folder_name}")

    while True:
        # Query for all items in this folder
        query = f"'{folder_id}' in parents and trashed=false"
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()

        items = response.get('files', [])

        for item in items:
            mime_type = item.get('mimeType')

            if mime_type == 'application/vnd.google-apps.folder':
                # Recursively process subfolder, using the subfolder's name as the new parent
                logger.info(f"Entering subfolder: {item['name']}")
                subfolder_files = list_all_files_recursively(
                    service,
                    item['id'],
                    item['name']  # Use the subfolder name for its files
                )
                files_list.extend(subfolder_files)
            else:
                # It's a file, add it with the current folder name (immediate parent)
                files_list.append({
                    'id': item['id'],
                    'name': item['name'],
                    'folder_name': current_folder_name,
                    'mimeType': mime_type
                })

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return files_list


def list_files_skip_root(service, folder_id: str) -> List[Dict]:
    """
    List all files from subfolders, ignoring the root folder level.
    Files are organized by their immediate parent subfolder name.

    Args:
        service: Google Drive API service
        folder_id: The root folder ID (this level will be ignored)

    Returns:
        List of file metadata dictionaries with 'id', 'name', 'folder_name', and 'mimeType'
    """
    files_list = []
    page_token = None

    root_folder_name = get_folder_name_by_id(service, folder_id)
    logger.info(f"Scanning root folder: {root_folder_name} (will be ignored)")

    # Get all items in the root folder
    while True:
        query = f"'{folder_id}' in parents and trashed=false"
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()

        items = response.get('files', [])

        for item in items:
            mime_type = item.get('mimeType')

            # Only process folders at the root level
            if mime_type == 'application/vnd.google-apps.folder':
                subfolder_name = item['name']
                logger.info(f"Processing subfolder: {subfolder_name}")

                # List all files in this subfolder (non-recursively, only direct children)
                subfolder_files = list_files_in_folder(service, item['id'], subfolder_name)
                files_list.extend(subfolder_files)
            else:
                # Skip files at the root level
                logger.debug(f"Skipping file at root level: {item['name']}")

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return files_list


def list_files_skip_two_levels(service, folder_id: str) -> List[Dict]:
    """
    List all files from tag folders (3rd level), ignoring the root and project folders.
    Traverses: Root -> Projects -> Tags -> Files
    Files are tagged with their tag folder name only.

    Args:
        service: Google Drive API service
        folder_id: The root folder ID

    Returns:
        List of file metadata dictionaries with 'id', 'name', 'folder_name' (tag), and 'mimeType'
    """
    files_list = []

    root_folder_name = get_folder_name_by_id(service, folder_id)
    logger.info(f"Scanning root folder: {root_folder_name}")

    # Level 1: Get all project folders
    page_token = None
    while True:
        query = f"'{folder_id}' in parents and trashed=false"
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()

        project_folders = response.get('files', [])

        for project in project_folders:
            if project['mimeType'] == 'application/vnd.google-apps.folder':
                logger.info(f"Scanning project: {project['name']}")

                # Level 2: Get all tag folders in this project
                tag_page_token = None
                while True:
                    tag_query = f"'{project['id']}' in parents and trashed=false"
                    tag_response = service.files().list(
                        q=tag_query,
                        spaces='drive',
                        fields='nextPageToken, files(id, name, mimeType)',
                        pageToken=tag_page_token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True
                    ).execute()

                    tag_folders = tag_response.get('files', [])

                    for tag in tag_folders:
                        if tag['mimeType'] == 'application/vnd.google-apps.folder':
                            logger.info(f"  Processing tag: {tag['name']}")

                            # Level 3: Get all files in this tag folder (recursively)
                            tag_files = list_all_files_recursively(
                                service,
                                tag['id'],
                                tag['name']
                            )
                            files_list.extend(tag_files)

                    tag_page_token = tag_response.get('nextPageToken')
                    if not tag_page_token:
                        break

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return files_list


def list_files_in_folder(service, folder_id: str, folder_name: str) -> List[Dict]:
    """
    List all files directly in a folder (non-recursive).

    Args:
        service: Google Drive API service
        folder_id: The folder ID
        folder_name: The folder name to tag files with

    Returns:
        List of file metadata dictionaries
    """
    files_list = []
    page_token = None

    while True:
        query = f"'{folder_id}' in parents and trashed=false"
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()

        items = response.get('files', [])

        for item in items:
            mime_type = item.get('mimeType')

            # Only include files (not folders)
            if mime_type != 'application/vnd.google-apps.folder':
                files_list.append({
                    'id': item['id'],
                    'name': item['name'],
                    'folder_name': folder_name,
                    'mimeType': mime_type
                })

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return files_list


def download_pdfs_skip_root(
    folder_url: str,
    download_dir: str = 'downloaded_pdfs',
    credentials_file: str = 'client_secret_833344150609-9b58bca2au71apgs3uiv0dbkif0oobgn.apps.googleusercontent.com.json'
) -> Dict[str, int]:
    """
    Download only PDF files from subfolders, ignoring the root folder.
    Files are renamed as subfolder_name_X.pdf where X is sequential per subfolder.

    Example:
        Root/АПЗ/doc1.pdf -> АПЗ_1.pdf
        Root/АПЗ/doc2.pdf -> АПЗ_2.pdf
        Root/ПЗУ/doc1.pdf -> ПЗУ_1.pdf

    Args:
        folder_url: Google Drive folder URL or folder ID (root folder will be ignored)
        download_dir: Directory to download files to
        credentials_file: Path to OAuth credentials file

    Returns:
        Dictionary with download statistics
    """
    # Authenticate
    logger.info("Authenticating with Google Drive...")
    service = authenticate_google_drive(credentials_file)

    # Extract folder ID from URL
    folder_id = extract_folder_id_from_url(folder_url)
    logger.info(f"Folder ID: {folder_id}")

    # Create download directory
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Download directory: {download_path.absolute()}")

    # List all files, skipping root folder
    logger.info("Scanning folder structure (ignoring root folder)...")
    all_files = list_files_skip_root(service, folder_id)
    logger.info(f"Found {len(all_files)} files total")

    # Filter only PDFs
    pdf_files = [f for f in all_files if f['name'].lower().endswith('.pdf')]
    logger.info(f"Found {len(pdf_files)} PDF files")

    # Group PDFs by folder name
    files_by_folder = defaultdict(list)
    for file_info in pdf_files:
        files_by_folder[file_info['folder_name']].append(file_info)

    # Download statistics
    stats = {
        'total_files': len(all_files),
        'total_pdfs': len(pdf_files),
        'successful': 0,
        'failed': 0
    }

    # Download PDFs
    for folder_name, files in files_by_folder.items():
        logger.info(f"\nProcessing {len(files)} PDFs from folder: {folder_name}")

        for idx, file_info in enumerate(files, 1):
            # Create new filename: foldername_number.pdf
            new_filename = f"{folder_name}_{idx}.pdf"
            local_path = download_path / new_filename

            logger.info(f"  [{idx}/{len(files)}] Downloading: {file_info['name']} -> {new_filename}")

            try:
                # Download file
                request = service.files().get_media(fileId=file_info['id'])
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        logger.debug(f"    Download {int(status.progress() * 100)}%")

                # Write to file
                try:
                    with open(local_path, 'wb') as f:
                        f.write(fh.getvalue())
                    stats['successful'] += 1
                    logger.info(f"    ✓ Saved to: {local_path}")
                except OSError as e:
                    logger.error(f"    ✗ Failed to write file {new_filename}: {e}")
                    logger.error(f"    Trying alternative filename...")
                    # Try with sanitized filename
                    safe_filename = f"{folder_name.replace('/', '_').replace('\\', '_')}_{idx}.pdf"
                    alt_path = download_path / safe_filename
                    with open(alt_path, 'wb') as f:
                        f.write(fh.getvalue())
                    stats['successful'] += 1
                    logger.info(f"    ✓ Saved to: {alt_path}")

            except Exception as e:
                logger.error(f"    ✗ Failed to download {file_info['name']}: {e}")
                stats['failed'] += 1

    return stats


def download_google_drive_files(
    folder_url: str,
    download_dir: str = 'tagged_documents',
    credentials_file: str = 'client_secret_833344150609-9b58bca2au71apgs3uiv0dbkif0oobgn.apps.googleusercontent.com.json'
) -> Dict[str, int]:
    """
    Download all files from a Google Drive folder recursively and rename them
    based on their shallowest parent folder name with enumeration.

    Args:
        folder_url: Google Drive folder URL or folder ID
        download_dir: Directory to download files to
        credentials_file: Path to OAuth credentials file

    Returns:
        Dictionary with download statistics
    """
    # Authenticate
    logger.info("Authenticating with Google Drive...")
    service = authenticate_google_drive(credentials_file)

    # Extract folder ID from URL
    folder_id = extract_folder_id_from_url(folder_url)
    logger.info(f"Folder ID: {folder_id}")

    # Create download directory
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Download directory: {download_path.absolute()}")

    # List all files recursively
    logger.info("Scanning folder structure...")
    all_files = list_all_files_recursively(service, folder_id)
    logger.info(f"Found {len(all_files)} files total")

    # Group files by folder name to handle enumeration
    files_by_folder = defaultdict(list)
    for file_info in all_files:
        files_by_folder[file_info['folder_name']].append(file_info)

    # Download files
    stats = {
        'total_files': len(all_files),
        'successful': 0,
        'failed': 0,
        'skipped_google_docs': 0
    }

    for folder_name, files in files_by_folder.items():
        logger.info(f"\nProcessing {len(files)} files from folder: {folder_name}")

        for idx, file_info in enumerate(files, 1):
            # Skip Google Docs native formats (Docs, Sheets, Slides)
            if file_info['mimeType'].startswith('application/vnd.google-apps.'):
                logger.warning(f"  Skipping Google native format: {file_info['name']} ({file_info['mimeType']})")
                stats['skipped_google_docs'] += 1
                continue

            # Get file extension
            original_name = file_info['name']
            file_extension = Path(original_name).suffix

            # Create new filename: foldername_number.ext
            new_filename = f"{folder_name}_{idx}{file_extension}"
            local_path = download_path / new_filename

            logger.info(f"  [{idx}/{len(files)}] Downloading: {original_name} -> {new_filename}")

            try:
                # Download file
                request = service.files().get_media(fileId=file_info['id'])
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        logger.debug(f"    Download {int(status.progress() * 100)}%")

                # Write to file with proper encoding handling
                try:
                    with open(local_path, 'wb') as f:
                        f.write(fh.getvalue())
                    stats['successful'] += 1
                    logger.info(f"    ✓ Saved to: {local_path}")
                except OSError as e:
                    logger.error(f"    ✗ Failed to write file {new_filename}: {e}")
                    logger.error(f"    Trying alternative filename...")
                    # Try with sanitized filename
                    safe_filename = f"{folder_name.replace('/', '_').replace('\\', '_')}_{idx}{file_extension}"
                    alt_path = download_path / safe_filename
                    with open(alt_path, 'wb') as f:
                        f.write(fh.getvalue())
                    stats['successful'] += 1
                    logger.info(f"    ✓ Saved to: {alt_path}")

            except Exception as e:
                logger.error(f"    ✗ Failed to download {original_name}: {e}")
                stats['failed'] += 1

    return stats


def load_download_status(csv_path: Path) -> Set[str]:
    """
    Load previously downloaded file IDs from CSV.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Set of file IDs that have been successfully downloaded
    """
    downloaded_ids = set()

    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['status'] == 'success':
                    downloaded_ids.add(row['file_id'])
        logger.info(f"Loaded {len(downloaded_ids)} previously downloaded files from {csv_path}")
    else:
        logger.info(f"No existing download status file found at {csv_path}")

    return downloaded_ids


def save_download_record(csv_path: Path, file_id: str, tag_name: str, original_name: str,
                         new_filename: str, status: str, error_msg: str = '', tag_short_name: str = ''):
    """
    Append a download record to the CSV file.

    Args:
        csv_path: Path to the CSV file
        file_id: Google Drive file ID
        tag_name: Tag folder name (full)
        original_name: Original filename on Drive
        new_filename: Local filename
        status: 'success' or 'failed'
        error_msg: Error message if failed
        tag_short_name: Shortened tag name used in filename
    """
    file_exists = csv_path.exists()

    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        fieldnames = ['file_id', 'tag_name', 'tag_short_name', 'original_name', 'new_filename', 'status', 'error_msg']
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'file_id': file_id,
            'tag_name': tag_name,
            'tag_short_name': tag_short_name,
            'original_name': original_name,
            'new_filename': new_filename,
            'status': status,
            'error_msg': error_msg
        })


def create_short_tag_name(tag_name: str, max_length: int = 80) -> str:
    """
    Create a shortened tag name suitable for filenames.
    If tag is too long, truncate and add hash suffix.

    Args:
        tag_name: Original tag name
        max_length: Maximum length for the short name

    Returns:
        Shortened tag name
    """
    if len(tag_name) <= max_length:
        return tag_name

    # Create a hash of the full name
    hash_suffix = hashlib.md5(tag_name.encode('utf-8')).hexdigest()[:8]

    # Truncate and add hash
    truncated = tag_name[:max_length - 9]  # Leave room for underscore and hash
    return f"{truncated}_{hash_suffix}"


def download_pdfs_by_tags(
    folder_url: str,
    download_dir: str = 'downloaded_pdfs',
    credentials_file: str = 'client_secret_833344150609-9b58bca2au71apgs3uiv0dbkif0oobgn.apps.googleusercontent.com.json',
    status_csv: str = 'download_status.csv'
) -> Dict[str, int]:
    """
    Download PDF files from tag folders (skipping root and project levels).
    Files from all projects are combined and renamed as tag_X.pdf where X is sequential per tag.
    Tracks download status in CSV and skips already downloaded files.

    Structure expected: Root -> Projects -> Tags -> Files
    Output: tag1_1.pdf, tag1_2.pdf, tag2_1.pdf, etc.

    Args:
        folder_url: Google Drive folder URL or folder ID
        download_dir: Directory to download files to
        credentials_file: Path to OAuth credentials file
        status_csv: Path to CSV file tracking download status

    Returns:
        Dictionary with download statistics
    """
    # Authenticate
    logger.info("Authenticating with Google Drive...")
    service = authenticate_google_drive(credentials_file)

    # Extract folder ID from URL
    folder_id = extract_folder_id_from_url(folder_url)
    logger.info(f"Folder ID: {folder_id}")

    # Create download directory
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Download directory: {download_path.absolute()}")

    # Load download status
    csv_path = Path(status_csv)
    downloaded_ids = load_download_status(csv_path)

    # List all files, skipping root and project folders
    logger.info("Scanning folder structure (Root -> Projects -> Tags -> Files)...")
    all_files = list_files_skip_two_levels(service, folder_id)
    logger.info(f"Found {len(all_files)} files total")

    # Filter only PDFs
    pdf_files = [f for f in all_files if f['name'].lower().endswith('.pdf')]
    logger.info(f"Found {len(pdf_files)} PDF files")

    # Filter out already downloaded files
    pdf_files_to_download = [f for f in pdf_files if f['id'] not in downloaded_ids]
    skipped_count = len(pdf_files) - len(pdf_files_to_download)

    if skipped_count > 0:
        logger.info(f"Skipping {skipped_count} already downloaded files")
    logger.info(f"Will download {len(pdf_files_to_download)} new PDF files")

    # Group PDFs by tag name (folder_name)
    files_by_tag = defaultdict(list)
    for file_info in pdf_files_to_download:
        files_by_tag[file_info['folder_name']].append(file_info)

    # Download statistics
    stats = {
        'total_files': len(all_files),
        'total_pdfs': len(pdf_files),
        'skipped': skipped_count,
        'successful': 0,
        'failed': 0
    }

    # Download PDFs
    for tag_name, files in files_by_tag.items():
        logger.info(f"\nProcessing {len(files)} PDFs from tag: {tag_name}")

        # Create short tag name for filesystem
        tag_short = create_short_tag_name(tag_name)
        if tag_short != tag_name:
            logger.info(f"  Using shortened tag name: {tag_short}")

        # Count existing files for this tag to continue enumeration
        existing_count = sum(1 for f in pdf_files if f['folder_name'] == tag_name and f['id'] in downloaded_ids)
        start_idx = existing_count + 1

        for idx, file_info in enumerate(files, start_idx):
            # Create new filename using short tag name: tagname_number.pdf
            new_filename = f"{tag_short}_{idx}.pdf"
            local_path = download_path / new_filename

            logger.info(f"  [{idx - start_idx + 1}/{len(files)}] Downloading: {file_info['name']} -> {new_filename}")

            try:
                # Download file
                request = service.files().get_media(fileId=file_info['id'])
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        logger.debug(f"    Download {int(status.progress() * 100)}%")

                # Write to file
                try:
                    with open(local_path, 'wb') as f:
                        f.write(fh.getvalue())
                    stats['successful'] += 1
                    logger.info(f"    ✓ Saved to: {local_path}")

                    # Record success
                    save_download_record(csv_path, file_info['id'], tag_name,
                                       file_info['name'], new_filename, 'success', '', tag_short)

                except OSError as e:
                    logger.error(f"    ✗ Failed to write file {new_filename}: {e}")
                    logger.error(f"    Trying alternative filename...")
                    # Try with sanitized filename
                    safe_tag = tag_short.replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_').replace('*', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                    safe_filename = f"{safe_tag}_{idx}.pdf"
                    alt_path = download_path / safe_filename
                    with open(alt_path, 'wb') as f:
                        f.write(fh.getvalue())
                    stats['successful'] += 1
                    logger.info(f"    ✓ Saved to: {alt_path}")

                    # Record success with sanitized filename
                    save_download_record(csv_path, file_info['id'], tag_name,
                                       file_info['name'], safe_filename, 'success', '', tag_short)

            except Exception as e:
                error_msg = str(e)
                logger.error(f"    ✗ Failed to download {file_info['name']}: {error_msg}")
                stats['failed'] += 1

                # Record failure
                save_download_record(csv_path, file_info['id'], tag_name,
                                   file_info['name'], new_filename, 'failed', error_msg, tag_short)

    return stats


if __name__ == "__main__":
    # Configuration
    FOLDER_URL = "https://drive.google.com/drive/folders/1XAMDNYey0pwntBYjWSWaDV1zX35G9NdF?usp=drive_link"

    logger.info("=" * 60)
    logger.info("Starting Google Drive PDF download")
    logger.info(f"Folder URL: {FOLDER_URL}")
    logger.info("Structure: Root -> Projects -> Tags -> Files")
    logger.info("Output: Files named by tag only (tag_1.pdf, tag_2.pdf, etc.)")
    logger.info("=" * 60)

    stats = download_pdfs_by_tags(
        folder_url=FOLDER_URL,
        download_dir='downloaded_pdfs'
    )

    logger.info("\n" + "=" * 60)
    logger.info("Download Summary:")
    logger.info(f"  Total files found: {stats['total_files']}")
    logger.info(f"  Total PDFs found: {stats['total_pdfs']}")
    logger.info(f"  Skipped (already downloaded): {stats['skipped']}")
    logger.info(f"  Successfully downloaded: {stats['successful']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info("=" * 60)
