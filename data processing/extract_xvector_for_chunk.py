# based on https://huggingface.co/speechbrain/spkrec-xvect-voxceleb

import os
import torch
import torchaudio
import pandas as pd
from speechbrain.pretrained import EncoderClassifier


# Load the pre-trained x-vector model
xvec_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_xvector"
)

torchaudio.set_audio_backend("soundfile") 

def extract_xvector_for_chunk(chunk_wav_path):
    try:
        # Load the audio file
        signal, sr = torchaudio.load(chunk_wav_path)
    except Exception as e:
        print(f"Error loading {chunk_wav_path}: {e}")
        return None
    # If the audio has multiple channels, convert it to mono by averaging
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    try:
        # Extract the x-vector, output shape [1, embedding_dim]
        embeddings = xvec_model.encode_batch(signal)
        xvec = embeddings.squeeze(0).detach().cpu().numpy()
        return xvec
    except Exception as e:
        print(f"Error extracting x-vector from {chunk_wav_path}: {e}")
        return None

def main():
    # load CSV file
    csv_file_path = "/home/wang.yan8/Aphasic_speech_recognition/final_clean_dataset.csv"
    df = pd.read_csv(csv_file_path)

    # Initialize a new column for storing x-vectors
    df["xvector"] = None

    # Iterate through each row, construct the audio file path, and extract x-vector
    for idx, row in df.iterrows():
        folder_name = row["folder_name"]
        chunk_file = row["file_cut"]
        audio_path = os.path.join("/home/wang.yan8/data_processed/audios", folder_name, chunk_file)
        
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            continue

        xvec = extract_xvector_for_chunk(audio_path)
        if xvec is None:
            print(f"Failed to extract x-vector for {audio_path}")
            continue
        
        # Convert the numpy array to a Python list, then to a string for CSV storage
        xvec_str = str(xvec.tolist())
        df.at[idx, "xvector"] = xvec_str
        print(f"Extracted x-vector for {chunk_file}")

    # Save the updated dataset with extracted x-vectors
    output_csv = "/home/wang.yan8/data_processed/final_clean_dataset_with_xvector.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved new CSV with x-vector to {output_csv}")

if __name__ == "__main__":
    main()
