# src/download_data.py
import os
import requests
import zipfile


def download_movielens_100k(data_dir="../data"):
    """
    Downloads the MovieLens 100K dataset from Grouplens and unzips it.
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-100k.zip")

    print(f"Downloading MovieLens 100K dataset to {zip_path}...")
    response = requests.get(url)
    with open(zip_path, "wb") as f:
        f.write(response.content)

    print(f"Unzipping dataset into {data_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print("Download complete.")


if __name__ == "__main__":
    download_movielens_100k()
