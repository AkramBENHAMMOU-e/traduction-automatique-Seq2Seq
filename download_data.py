import os
import zipfile

# Set Kaggle config dir to current directory so it finds kaggle.json here
os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

import kaggle


DATASET_NAME = "devicharith/language-translation-englishfrench"
DOWNLOAD_PATH = "data/"

def download_and_extract():
    if not os.path.exists("data"):
        os.makedirs("data")

    print(f"Authenticating with Kaggle using config in {os.getcwd()}...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset {DATASET_NAME}...")
    api.dataset_download_files(DATASET_NAME, path=DOWNLOAD_PATH, unzip=True)
    print("Download and extraction complete.")

if __name__ == "__main__":
    download_and_extract()
