import os
import requests
import shutil
from tqdm import tqdm

def download_glove(dimension):
    base_url = "https://nlp.stanford.edu/data/"
    file_name = f"glove.6B.{dimension}d.zip"
    url = base_url + file_name

    print(f"Downloading GloVe {dimension}d embeddings...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(file_name, 'wb') as file, tqdm(
        desc=file_name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

    print(f"Download complete. File saved as {file_name}")

    # Source file/directory path
    source = file_name

    # Destination directory path
    destination = "data/glove/"

    if not os.path.exists('data'):
        os.mkdir('data')
        os.mkdir('data/glove')
    # Move the file 
    shutil.move(source, destination) 
    

def main():
    print("Available GloVe dimensions:")
    print("1. 50d")
    print("2. 100d")
    print("3. 200d")
    print("4. 300d")

    choice = input("Enter the number of the GloVe dimension you want to download (1-4): ")

    dimension_map = {
        '1': '50',
        '2': '100',
        '3': '200',
        '4': '300'
    }

    if choice in dimension_map:
        dimension = dimension_map[choice]
        download_glove(dimension)
    else:
        print("Invalid choice. Please run the script again and select a number between 1 and 4.")

if __name__ == "__main__":
    main()