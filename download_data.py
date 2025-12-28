import argparse
import os
import zipfile
from urllib.request import Request, urlopen


TATOEBA_FRA_ENG_URL = "http://www.manythings.org/anki/fra-eng.zip"
DEFAULT_OUT_DIR = os.path.join("data", "tatoeba")
DEFAULT_ZIP_PATH = os.path.join(DEFAULT_OUT_DIR, "fra-eng.zip")
DEFAULT_TXT_PATH = os.path.join(DEFAULT_OUT_DIR, "fra.txt")


def _download(url: str, dst_path: str) -> None:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(req) as resp, open(dst_path, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def download_and_extract(url: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, os.path.basename(url))

    if not os.path.exists(zip_path):
        print(f"Downloading {url} -> {zip_path}")
        _download(url, zip_path)
    else:
        print(f"Zip already exists: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    extracted_txt = os.path.join(out_dir, "fra.txt")
    if not os.path.exists(extracted_txt):
        raise FileNotFoundError(
            f"Expected {extracted_txt} after extracting {zip_path}. "
            "The zip contents may have changed."
        )

    print(f"OK: extracted dataset to {extracted_txt}")
    return extracted_txt


def main():
    parser = argparse.ArgumentParser(description="Download the Tatoeba (ManyThings) English-French dataset.")
    parser.add_argument("--url", type=str, default=TATOEBA_FRA_ENG_URL)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    download_and_extract(args.url, args.out_dir)


if __name__ == "__main__":
    main()
