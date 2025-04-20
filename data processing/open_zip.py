import zipfile
import sys
import os

def open_zip_file(file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 open_zip.py <path_to_zip_file> <output_directory>")
        sys.exit(1)

    zip_file_path = sys.argv[1]
    output_directory = sys.argv[2]
    open_zip_file(zip_file_path, output_directory)