import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path
import time
import zipfile


class ZenodoScraper:
    def __init__(self, record_url, download_dir="dataset"):
        self.record_url = record_url
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })

    def get_file_links(self):
        print(f"Fetching page: {self.record_url}")
        response = self.session.get(self.record_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all download links
        files = []

        file_elements = soup.find_all('a', href=True)

        for elem in file_elements:
            href = elem['href']
            if '/files/' in href and not href.endswith('/files/'):
                filename = href.split('/files/')[-1]
                if filename:
                    if href.startswith('/'):
                        full_url = f"https://zenodo.org{href}"
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue

                    files.append({
                        'filename': filename,
                        'url': full_url
                    })

        seen = set()
        unique_files = []
        for f in files:
            if f['filename'] not in seen:
                seen.add(f['filename'])
                unique_files.append(f)

        return unique_files

    def unzip_file(self, zip_path, extract_to=None):
        """Unzip a file to the specified directory."""
        if extract_to is None:
            extract_to = zip_path.parent / zip_path.stem

        extract_to.mkdir(exist_ok=True)

        print(f"Extracting: {zip_path.name}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files to extract for progress bar
                file_list = zip_ref.namelist()

                with tqdm(total=len(file_list), desc="Extracting", unit="file") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, extract_to)
                        pbar.update(1)

            print(f"Successfully extracted to: {extract_to}")
            return extract_to

        except zipfile.BadZipFile:
            print(f"Error: {zip_path.name} is not a valid zip file")
            return None
        except Exception as e:
            print(f"Error extracting {zip_path.name}: {e}")
            return None

    def download_file(self, url, filename, chunk_size=8192, auto_unzip=True):
        """Download a file with progress bar and optionally unzip it."""
        filepath = self.download_dir / filename

        if filepath.exists():
            print(f"{filename} already exists, skipping download...")
            # Still try to unzip if it's a zip file and auto_unzip is enabled
            if auto_unzip and filename.lower().endswith('.zip'):
                extract_dir = filepath.parent / filepath.stem
                if not extract_dir.exists():
                    self.unzip_file(filepath)
            return

        print(f"\nDownloading: {filename}")
        print(f"URL: {url}")

        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=filename
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            print(f"Successfully downloaded: {filename}")

            # Auto-unzip if it's a zip file
            if auto_unzip and filename.lower().endswith('.zip'):
                self.unzip_file(filepath)

        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            if filepath.exists():
                filepath.unlink()

    def download_all(self, file_filter=None, auto_unzip=True):
        """Download all files from the Zenodo record.

        Args:
            file_filter: Optional list of filenames to download.
                        If None, downloads all files.
            auto_unzip: If True, automatically unzip .zip files after download.
                       Default is True.
        """
        files = self.get_file_links()

        if not files:
            print("No files found on the page!")
            return

        print(f"\nFound {len(files)} file(s):")
        for f in files:
            print(f"  - {f['filename']}")

        # Filter files if requested
        if file_filter:
            files = [f for f in files if f['filename'] in file_filter]
            print(f"\nFiltering to {len(files)} file(s)")

        print(f"\nDownload directory: {self.download_dir.absolute()}")

        for i, file_info in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]")
            self.download_file(file_info['url'], file_info['filename'], auto_unzip=auto_unzip)
            time.sleep(0.5)  # Be nice to the server

        print("Download complete!")


def main():
    # Configuration
    RECORD_URL = "https://zenodo.org/records/6998231"
    DOWNLOAD_DIR = "genea_dataset"

    scraper = ZenodoScraper(RECORD_URL, DOWNLOAD_DIR)

    scraper.download_all()

    # Option 2: Download specific files only (uncomment to use)
    # scraper.download_all(file_filter=['README.txt', 'LICENSE.txt', 'val.zip'])


if __name__ == "__main__":
    main()
