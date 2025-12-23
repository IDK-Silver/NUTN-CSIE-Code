import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# Dataset URL and file configuration
URL = "https://nutn-code.yuufeng.com/data-mining/114-1-dm-final-project.zip"
FILENAME = "114-1-dm-final-project.zip"

# Directory paths
BLOB_DIR = Path("blobs")
UNZIP_DIR = BLOB_DIR / "unzip"
RAW_DIR = BLOB_DIR / "raw"

# Create necessary directories if they don't exist
UNZIP_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Full path for the downloaded zip file
zip_path = UNZIP_DIR / FILENAME

# Download the dataset with progress bar
print(f"Downloading dataset from {URL}...")
response = requests.get(URL, stream=True)
response.raise_for_status()

# Get total file size for progress bar
total_size = int(response.headers.get('content-length', 0))
block_size = 1024 * 1024  # 1MB chunks

# Download with progress bar
with open(zip_path, "wb") as f:
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=FILENAME) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

print(f"Downloaded to: {zip_path}")

# Extract the zip file to the raw directory
print(f"\nExtracting to {RAW_DIR}...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Get list of files in the archive for progress bar
    file_list = zip_ref.namelist()

    # Extract with progress bar
    with tqdm(total=len(file_list), desc="Extracting", unit="file") as pbar:
        for file in file_list:
            zip_ref.extract(file, RAW_DIR)
            pbar.update(1)

print(f"Extraction complete!")
print(f"ZIP file kept at: {zip_path}")
print(f"Extracted files at: {RAW_DIR}")
