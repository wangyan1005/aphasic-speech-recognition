# Aphasic Speech Recognition

## Overview
This project focuses on aphasic speech recognition using the AphasiaBank dataset and the Whisper model. The goal is to preprocess and fine-tune speech recognition models to better understand and transcribe speech from individuals with aphasia.

## Dataset
- **AphasiaBank Dataset**: A collection of audio recordings and transcriptions from individuals with aphasia, widely used for research in speech and language processing.
- The dataset provides metadata such as patient names, age, sex, and conversation transcriptions.

## Model
- **Whisper Model**: A state-of-the-art speech recognition model developed by OpenAI.
- The model is fine-tuned using LoRA (Low-Rank Adaptation) to efficiently adapt to aphasic speech patterns.

## Data Processing
The data preprocessing is based on https://github.com/Liting-Zhou/Aphasic_speech_recognition.
I have added new files:
- **test.py**:The provided scripts test if the dataset has non-standard audio segments.
- **hours_analysis.py**: The scripts extract and analyze the total speech duration for each speaker.
- **extract_xvector_for_chunk.py**: extract xvector from audio chunks
- **data_splitting.py**: Split the dataset into training (60%), validation (20%), and test (20%) sets

## Baseline Model Transcribe
- **transcribe.py**: The script transcribes the test set audio using Whisper model from Hugging Face's pipeline, and writes the detailed transcription results (including predictions and references) to a CSV file.
- **wer_calculation.py**: This script reads the detailed transcription results and calculates the overall Word Error Rate (WER).

## Training Models
- **data_preparation.py**: Prepares and processes the dataset by reading a CSV file, loading audio files, extracting log-Mel features, and parsing x-vectors.
- **data_collator.py**: Provides a data collator that pads input features and labels for speech recognition, and also processes x-vectors.
- **personalized_whisper.py**: Implements the personalized Whisper model. This module integrates speaker-specific x-vector information with the log-Mel spectrogram features.
- **compute_metrics.py**: Defines functions to compute the Word Error Rate (WER).
- **training.py**: Contains the main training script. It loads the prepared dataset, applies LoRA to the base Whisper model (e.g., Whisper-Small).

### Run Scripts Summary
1. **Data Preparation**:
 ```bash
   python data_preparation.py
 ```
2. **Training**:
```bash
   python training.py --lora_rank 8
```



  
