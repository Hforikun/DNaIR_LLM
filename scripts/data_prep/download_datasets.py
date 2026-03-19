import os
import urllib.request
import urllib.error
import zipfile
import io
import shutil

def download_and_extract(url, extract_dir, subfolder=None):
    print(f"Downloading from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                z.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")
        if subfolder:
            print(f"Checking {subfolder} structure...")
    except urllib.error.URLError as e:
        print(f"Failed to download {url}: {e}")

def main():
    base_dir = "/Users/kerwin/Desktop/RS/DNaIR-LLM"
    raw_dir = os.path.join(base_dir, "data/raw")
    
    # 1. MovieLens 100K
    # ml-100k.zip contains a folder 'ml-100k', so extracting to raw_dir is fine
    ml_url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    download_and_extract(ml_url, raw_dir)
    
    # 2. FilmTrust
    # We will extract to a temporary folder and then organize it into data/raw/filmtrust
    ft_url = "https://guoguibing.github.io/librec/datasets/filmtrust.zip"
    ft_extract_dir = os.path.join(raw_dir, "filmtrust_temp")
    os.makedirs(ft_extract_dir, exist_ok=True)
    download_and_extract(ft_url, ft_extract_dir)
    
    # Organize FilmTrust
    ft_dest = os.path.join(raw_dir, "filmtrust")
    os.makedirs(ft_dest, exist_ok=True)
    
    # Typically it extracts subfolders or files directly. Let's move files generically.
    for root, dirs, files in os.walk(ft_extract_dir):
        for file in files:
            if file.endswith('.txt') or file.endswith('.csv'):
                shutil.move(os.path.join(root, file), os.path.join(ft_dest, file))
    
    print("\nDataset download and extraction complete.")

if __name__ == "__main__":
    main()
