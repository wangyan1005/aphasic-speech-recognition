# based on https://github.com/Liting-Zhou/Aphasic_speech_recognition/blob/main/generate_audio_chunks.py
import os
import glob
import pandas as pd
from tqdm import tqdm  # Import tqdm for the progress bar
import sys
from io import StringIO
from pydub import AudioSegment


def process_audio_chunks(transcript_folder_path, audio_folder_path):
    filepath = os.path.join(transcript_folder_path, 'clean_dataset.csv')

    if not os.path.exists(filepath):
        print(f"No clean_dataset.csv found in {transcript_folder_path}, skipping...")
        return

    df = pd.read_csv(filepath)

    # Use tqdm for progress bar in processing the DataFrame rows
    for i in tqdm(range(len(df)), desc=f"Processing {transcript_folder_path}", unit="file"):
        file = glob.glob(os.path.join(audio_folder_path, f"""{df['file'][i]}"""), recursive=True)
        if not file:
            print(f"Audio file not found for {df['file'][i]}, skipping...")
            continue

        start = ((pd.to_numeric(df['mark_start'][i])) / 1000)
        duration = ((pd.to_numeric(df['mark_end'][i])) - (pd.to_numeric(df['mark_start'][i]))) / 1000
        output_file = f"{file[0][:-4]}_{start}_{duration}.wav"

        # revised for skipping the existed files
        if os.path.exists(output_file):
            print(f"Skipped (already processed): {output_file}")
            continue
        
        # Load the audio file using pydub
        audio = AudioSegment.from_wav(file[0])
        
        # Extract the desired chunk
        chunk = audio[start * 1000: (start + duration) * 1000]
        
        # Export the chunk as a new WAV file
        chunk.export(output_file, format="wav")

        print(f"Processed: {file[0]} -> {output_file}")

def process_all_folders(root_dir):
    # Use tqdm for progress bar in processing all folders
    all_folders = [dir_name for root, dirs, files in os.walk(root_dir) for dir_name in dirs]
    
    for dir_name in tqdm(all_folders, desc="Processing folders", unit="folder"):
        transcript_folder_path = os.path.join(root_dir, dir_name)
        audio_folder_path = transcript_folder_path.replace("transcripts", "audios")
        print(f"Processing folder: {audio_folder_path}")
        process_audio_chunks(transcript_folder_path, audio_folder_path)

################################################

transcripts_dir = "../data_processed/transcripts"
process_all_folders(transcripts_dir)