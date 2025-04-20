# based on https://github.com/monirome/AphasiaBank/blob/main/convert_mp4_to_wav.py

import os
import glob
import sys

def convert_audio(input_path, output_path):

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Convert mp4 format to wav format
    mp4_files = glob.glob(input_path + "/*.mp4", recursive=True)
    if mp4_files:
        for file in mp4_files:
            output_file = os.path.join(output_path, os.path.basename(file)[:-4] + ".wav")
            if os.path.exists(output_file):
                print(f"Skipping {output_file}, already exists.")
                continue
            # Convert mp4 to wav using ffmpeg
            os.system(f"""ffmpeg -i "{file}" -ar 16000 -ac 1 "{output_file}" """)

    # Convert mp3 format to wav format
    mp3_files = glob.glob(input_path + "/*.mp3", recursive=True)
    if mp3_files:
        for file in mp3_files:
            output_file = os.path.join(output_path, os.path.basename(file)[:-4] + ".wav")
            if os.path.exists(output_file):
                print(f"Skipping {output_file}, already exists.")
                continue
            # Convert mp3 to wav using ffmpeg
            os.system(f"""ffmpeg -i "{file}" -ar 16000 -ac 1 "{output_file}" """)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_to_wav.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_audio(input_path, output_path)