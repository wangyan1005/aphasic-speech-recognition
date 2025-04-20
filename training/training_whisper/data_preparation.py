# based on https://huggingface.co/blog/fine-tune-whisper

import argparse
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
import soundfile as sf
from datasets import Dataset, DatasetDict, Audio,load_from_disk
import pandas as pd
import os 
import ast  # Used to convert strings into lists

def process_dataset():
    # fix the model size
    model_name = "openai/whisper-small"

    # load dataset
    csv_file_path = '/home/wang.yan8/data_processed/dataset_splitted.csv'
    df = pd.read_csv(csv_file_path)

    columns_to_drop = ['mark_start', 'mark_end', 'name','sex','age','file','WAB_AQ','aphasia_type','WAB_AQ_category','fluency_speech','original_file_length','difference','name_extracted_from_filename','name_unique_speaker']
    df = df.drop(columns=columns_to_drop)

    dataset = Dataset.from_pandas(df)

    dataset_dict=DatasetDict()
    dataset_dict["train"] = dataset.filter(lambda example: example["split"] == "train")
    dataset_dict["eval"] = dataset.filter(lambda example: example["split"] == "validation")
    dataset_dict["test"] = dataset.filter(lambda example: example["split"] == "test")

    print("Data splitting finished.")

    # directory to save the processed audio dataset
    processed_audio_data_path = f'/scratch/wang.yan8/processed_audio_dataset_small'

    # list to keep track of missing audio files
    missing_files = []

    def load_audio(batch):
        audio_file_path = os.path.join("/home/wang.yan8/data_processed/audios", batch["folder_name"], batch["file_cut"])
        # check if the file exists
        if os.path.exists(audio_file_path):
            try:
                audio, sample_rate = sf.read(audio_file_path)
                batch["audio"] = {"array": audio, "sampling_rate": sample_rate}
            except Exception as e:
                print(f"Error loading {audio_file_path}: {e}")
                batch["audio"] = None
        else:
            print(f"File not found: {audio_file_path}")
            missing_files.append(audio_file_path)
            batch["audio"] = None  # assign None for missing audio files
        
        return batch

    # check if the processed dataset already exists
    if os.path.exists(processed_audio_data_path):
        dataset_dict = load_from_disk(processed_audio_data_path)
        print("Loaded existing audio dataset!")
    else:
        dataset_dict = dataset_dict.map(load_audio)
        dataset_dict.save_to_disk(processed_audio_data_path)
        print("Processed audio dataset saved.")

    # save the missing files to a CSV
    if missing_files:
        missing_files_df = pd.DataFrame(missing_files, columns=["missing_file_path"])
        missing_files_df.to_csv("/home/wang.yan8/data_processed/missing_audio_files.csv", index=False)
        print("Missing audio files saved to 'missing_audio_files.csv'.")
    else:
        print("No missing audio files detected.")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="English", task="transcribe")

    def prepare_dataset(batch):
        audio = batch["audio"]
        if audio is None or not audio["array"]:
            print(f"Invalid or empty audio data for file: {batch.get('file_cut', 'unknown')}")
            batch["input_features"] = None
            batch["labels"] = None
            batch["xvector"] = [0.0] * 512 
            return batch  
    
        try:
            # compute log-Mel input features from input audio array
            batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

            # encode target text to label ids
            transcription = batch.get("transcriptions", "")
            if isinstance(transcription, str) and transcription.strip(): 
                batch["labels"] = tokenizer(transcription).input_ids
            else:
                print(f"Warning: Empty transcription for file {batch.get('file_cut', 'unknown')}")
                batch["labels"] = []
    
        except Exception as e:
            print(f"Error processing audio or transcription: {e}")
            batch["input_features"] = None
            batch["labels"] = []

        # Parse x-vector
        if "xvector" in batch and isinstance(batch["xvector"], str):
            try:
                parsed = ast.literal_eval(batch["xvector"])
                if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], list):
                    batch["xvector"] = parsed[0]
                else:
                    batch["xvector"] = parsed
            except Exception as e:
                print(f"Error parsing xvector for file {batch.get('file_cut', 'unknown')}: {e}")
                batch["xvector"] = [0.0] * 512  

        # Debug 
        print(f"Processed file {batch.get('file_cut', 'unknown')}: input_features={type(batch['input_features'])}, labels={batch['labels']}, xvector={len(batch['xvector'])}")
    
        return batch


    dataset_dict = dataset_dict.map(prepare_dataset, num_proc=6)

    print("finished preparing dataset")

    # save the dataset_dict
    dataset_dict_path = f'/scratch/wang.yan8/dataset_dict_small'
    dataset_dict.save_to_disk(dataset_dict_path)
    print("Dataset_dict saved to disk.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a dataset with Whisper model.")
    process_dataset()